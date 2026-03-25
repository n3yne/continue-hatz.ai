import {
  ChatMessage,
  CompletionOptions,
  LLMOptions,
  MessageContent,
} from "../../index.js";
import { BaseLLM } from "../index.js";
import { toChatBody } from "../openaiTypeConverters.js";

/**
 * Flatten content from array format to plain string.
 * Hatz API requires content to be a string, not an array of content parts.
 */
function flattenContent(content: MessageContent): string {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .filter(
        (part): part is { type: "text"; text: string } =>
          typeof part === "object" && part.type === "text",
      )
      .map((part) => part.text)
      .join("\n");
  }
  return String(content ?? "");
}

/**
 * Types for the OpenAI Responses API wire format
 */
interface ResponsesInputTextContent {
  type: "input_text";
  text: string;
}

interface ResponsesOutputTextContent {
  type: "output_text";
  text: string;
}

interface ResponsesUserMessage {
  role: "user";
  content: ResponsesInputTextContent[];
}

interface ResponsesSystemMessage {
  role: "system" | "developer";
  content: string;
}

interface ResponsesAssistantMessage {
  role: "assistant";
  content: ResponsesOutputTextContent[];
}

interface ResponsesFunctionCall {
  type: "function_call";
  id: string;
  call_id: string;
  name: string;
  arguments: string;
}

interface ResponsesFunctionCallOutput {
  type: "function_call_output";
  call_id: string;
  output: string;
}

type ResponsesInputItem =
  | ResponsesUserMessage
  | ResponsesSystemMessage
  | ResponsesAssistantMessage
  | ResponsesFunctionCall
  | ResponsesFunctionCallOutput;

interface ResponsesOutputFunctionCall {
  type: "function_call";
  id: string;
  call_id: string;
  name: string;
  arguments: string;
}

interface ResponsesOutputMessage {
  type: "message";
  role: "assistant";
  content: { type: "output_text"; text: string }[];
}

type ResponsesOutputItem = ResponsesOutputMessage | ResponsesOutputFunctionCall;

class HatzAI extends BaseLLM {
  static providerName = "hatz";
  static defaultOptions: Partial<LLMOptions> | undefined = {
    apiBase: "https://ai.hatz.ai/v1/",
    useLegacyCompletionsEndpoint: false,
  };

  public autoTool: boolean = false;

  public useLegacyCompletionsEndpoint: boolean = false;

  constructor(options: LLMOptions) {
    super(options);
    this.useLegacyCompletionsEndpoint = false;
    this.autoTool = (options as any).autoTool ?? false;
  }

  // Hatz does not support the legacy /completions endpoint
  supportsCompletions(): boolean {
    return false;
  }

  // Hatz does not support FIM
  supportsFim(): boolean {
    return false;
  }

  supportsTools(): "native" | "llm" | undefined {
    return "native";
  }

  protected _getHeaders() {
    return {
      "Content-Type": "application/json",
      ...(this.apiKey && { "X-API-Key": this.apiKey }),
      ...(this.apiKey && { Authorization: `Bearer ${this.apiKey}` }),
    };
  }

  protected _getEndpoint(
    endpoint: "chat/completions" | "completions" | "models" | "responses",
  ) {
    if (!this.apiBase) {
      throw new Error(
        "No API base URL provided. Please set the 'apiBase' option in config.yaml",
      );
    }
    // Hatz uses /chat/models instead of /models
    if (endpoint === "models") {
      return new URL("chat/models", this.apiBase);
    }
    return new URL(endpoint, this.apiBase);
  }

  /**
   * Get the OpenAI Responses API endpoint.
   * Per Hatz docs, this is at /v1/openai/responses
   */
  private _getResponsesEndpoint(): URL {
    if (!this.apiBase) {
      throw new Error(
        "No API base URL provided. Please set the 'apiBase' option in config.yaml",
      );
    }
    return new URL("openai/responses", this.apiBase);
  }

  /**
   * Convert Continue ChatMessage[] to OpenAI Responses API input format.
   *
   * Mapping:
   *  - system message     → { role: "system", content: "..." }
   *  - user message       → { role: "user", content: [{ type: "input_text", text: "..." }] }
   *  - assistant message  → { role: "assistant", content: [{ type: "output_text", text: "..." }] }
   *                          + separate function_call items for each toolCall
   *  - tool message       → { type: "function_call_output", call_id: "...", output: "..." }
   */
  private _convertMessagesToResponsesInput(
    messages: ChatMessage[],
  ): ResponsesInputItem[] {
    const input: ResponsesInputItem[] = [];

    for (const msg of messages) {
      const contentStr = flattenContent(msg.content);

      switch (msg.role) {
        case "system":
          input.push({
            role: "system",
            content: contentStr,
          });
          break;

        case "user":
          input.push({
            role: "user",
            content: [{ type: "input_text", text: contentStr }],
          });
          break;

        case "assistant":
          // Add assistant text content if present
          if (contentStr) {
            input.push({
              role: "assistant",
              content: [{ type: "output_text", text: contentStr }],
            });
          }
          // Add function_call items for each tool call
          if (msg.toolCalls && msg.toolCalls.length > 0) {
            for (const tc of msg.toolCalls) {
              input.push({
                type: "function_call",
                id: tc.id ?? `call_${Date.now()}`,
                call_id: tc.id ?? `call_${Date.now()}`,
                name: tc.function?.name ?? "",
                arguments: tc.function?.arguments ?? "{}",
              });
            }
          }
          break;

        case "tool": {
          const toolCallId =
            "toolCallId" in msg
              ? (msg as ChatMessage & { toolCallId: string }).toolCallId
              : "unknown";
          input.push({
            type: "function_call_output",
            call_id: toolCallId,
            output: contentStr,
          });
          break;
        }

        default:
          // For any other role, treat as user message
          input.push({
            role: "user",
            content: [{ type: "input_text", text: contentStr }],
          });
          break;
      }
    }

    return input;
  }

  /**
   * Parse OpenAI Responses API output items into ChatMessages.
   * Returns separate messages: one for text content, one for tool calls.
   */
  private _parseResponsesOutput(output: ResponsesOutputItem[]): ChatMessage[] {
    const results: ChatMessage[] = [];
    let textContent = "";
    const toolCalls: {
      id: string;
      type: "function";
      function: { name: string; arguments: string };
    }[] = [];

    for (const item of output) {
      if (item.type === "message" && item.content) {
        for (const part of item.content) {
          if (part.type === "output_text") {
            textContent += part.text;
          }
        }
      } else if (item.type === "function_call") {
        toolCalls.push({
          id: item.call_id ?? item.id,
          type: "function",
          function: {
            name: item.name,
            arguments: item.arguments,
          },
        });
      }
    }

    // Yield text content first if present
    if (textContent) {
      results.push({
        role: "assistant",
        content: textContent,
      });
    }

    // Yield tool calls as a separate message
    if (toolCalls.length > 0) {
      results.push({
        role: "assistant",
        content: "",
        toolCalls,
      });
    }

    return results;
  }

  /**
   * Use the OpenAI Responses API (/v1/openai/responses) for tool-enabled requests.
   * This endpoint returns structured function_call output items instead of
   * embedding tool calls as XML in the content string.
   */
  private async *_streamChatResponses(
    messages: ChatMessage[],
    signal: AbortSignal,
    options: CompletionOptions,
  ): AsyncGenerator<ChatMessage> {
    const input = this._convertMessagesToResponsesInput(messages);

    const requestBody: Record<string, unknown> = {
      model: options.model ?? this.model,
      input,
      stream: false,
    };

    // Include tools
    if (options.tools && options.tools.length > 0) {
      requestBody.tools = options.tools.map(
        (tool: {
          type: string;
          function: {
            name: string;
            description?: string;
            parameters?: unknown;
          };
        }) => ({
          type: "function",
          name: tool.function.name,
          description: tool.function.description ?? "",
          parameters: tool.function.parameters ?? {},
        }),
      );
    }

    // Include optional parameters
    if (options.temperature !== undefined) {
      requestBody.temperature = options.temperature;
    }
    if (options.topP !== undefined) {
      requestBody.top_p = options.topP;
    }
    if (options.maxTokens !== undefined) {
      requestBody.max_output_tokens = options.maxTokens;
    }

    const response = await this.fetch(this._getResponsesEndpoint(), {
      method: "POST",
      headers: this._getHeaders(),
      body: JSON.stringify(requestBody),
      signal,
    });

    if ((response as any).status === 499) {
      return;
    }

    if ((response as any).status >= 400) {
      const errorText = await response.text();
      throw new Error(
        `Hatz Responses API error ${(response as any).status}: ${errorText}`,
      );
    }

    const data = await response.json();

    if (data.output && Array.isArray(data.output)) {
      const chatMessages = this._parseResponsesOutput(
        data.output as ResponsesOutputItem[],
      );
      for (const chatMessage of chatMessages) {
        yield chatMessage;
      }
    } else if (data.choices?.[0]?.message) {
      // Fallback in case the API returns chat completions format
      const message = data.choices[0].message;
      const chatMessage: ChatMessage = {
        role: "assistant",
        content: message.content ?? "",
      };
      if (message.tool_calls && message.tool_calls.length > 0) {
        chatMessage.toolCalls = message.tool_calls.map(
          (tc: {
            id: string;
            type: string;
            function: { name: string; arguments: string };
          }) => ({
            id: tc.id,
            type: tc.type as "function",
            function: {
              name: tc.function.name,
              arguments: tc.function.arguments,
            },
          }),
        );
      }
      yield chatMessage;
    }
  }

  /**
   * Convert messages to Hatz-compatible format for /chat/completions.
   * Flattens all content arrays to plain strings.
   */
  private _convertMessagesForHatz(body: any): Record<string, unknown> {
    const messages = (body.messages as ChatMessage[]).map((msg) => {
      const converted: Record<string, unknown> = {
        role: msg.role,
        content: flattenContent(msg.content),
      };
      return converted;
    });

    const hatzBody: Record<string, unknown> = {
      ...body,
      messages,
      stream: (body.stream as boolean) ?? false,
      ...(this.autoTool && { auto_tool: true }),
    };

    return hatzBody;
  }

  /**
   * Use /chat/completions for non-tool requests (regular conversation).
   */
  private async *_streamChatCompletions(
    messages: ChatMessage[],
    signal: AbortSignal,
    options: CompletionOptions,
  ): AsyncGenerator<ChatMessage> {
    const body = toChatBody(messages, options, {});
    const hatzBody = this._convertMessagesForHatz(body);
    hatzBody.stream = false;

    const response = await this.fetch(this._getEndpoint("chat/completions"), {
      method: "POST",
      headers: this._getHeaders(),
      body: JSON.stringify(hatzBody),
      signal,
    });

    if ((response as any).status === 499) {
      return;
    }

    if ((response as any).status >= 400) {
      const errorText = await response.text();
      throw new Error(
        `Hatz API error ${(response as any).status}: ${errorText}`,
      );
    }

    const data = await response.json();

    if (data.choices?.[0]?.message) {
      const message = data.choices[0].message;
      yield {
        role: "assistant",
        content: message.content ?? "",
      };
    }
  }

  /**
   * Route to the appropriate API based on whether tools are present.
   * - With tools: use /v1/openai/responses (client-managed tool calling)
   * - Without tools: use /v1/chat/completions (regular conversation)
   */
  protected async *_streamChat(
    messages: ChatMessage[],
    signal: AbortSignal,
    options: CompletionOptions,
  ): AsyncGenerator<ChatMessage> {
    if (options.tools && options.tools.length > 0) {
      yield* this._streamChatResponses(messages, signal, options);
    } else {
      yield* this._streamChatCompletions(messages, signal, options);
    }
  }

  protected async _complete(
    prompt: string,
    signal: AbortSignal,
    options: CompletionOptions,
  ): Promise<string> {
    let completion = "";
    for await (const chunk of this._streamChat(
      [{ role: "user", content: prompt }],
      signal,
      options,
    )) {
      completion += chunk.content;
    }
    return completion;
  }

  protected async *_streamComplete(
    prompt: string,
    signal: AbortSignal,
    options: CompletionOptions,
  ): AsyncGenerator<string> {
    for await (const chunk of this._streamChat(
      [{ role: "user", content: prompt }],
      signal,
      options,
    )) {
      yield typeof chunk.content === "string" ? chunk.content : "";
    }
  }

  async listModels(): Promise<string[]> {
    const response = await this.fetch(this._getEndpoint("models"), {
      method: "GET",
      headers: this._getHeaders(),
    });

    const data = await response.json();
    // Hatz returns { data: [{ name: "model-id", display_name: "...", ... }] }
    return data.data.map((m: any) => m.name);
  }
}

export default HatzAI;
