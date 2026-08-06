/**
 * genai-optimizer JavaScript client.
 *
 * Start the service first:
 *     genai-optimize serve --port 8088
 *
 * Then from Node, Next.js, or the browser:
 *
 *     import { GenAIOptimizer } from "./optimizer.js";
 *     const opt = new GenAIOptimizer("http://localhost:8088");
 *     const r = await opt.optimizePrompt("Could you please kindly help me...");
 *     console.log(r.optimized_prompt, r.percent_saved);
 *
 * Works with plain fetch, no dependency.
 */

export class GenAIOptimizer {
  constructor(baseUrl = "http://localhost:8088", options = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.timeout = options.timeout || 15000;
    this.headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  }

  async _post(path, body) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);
    try {
      const res = await fetch(this.baseUrl + path, {
        method: "POST",
        headers: this.headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || `optimizer returned ${res.status}`);
      }
      return data;
    } finally {
      clearTimeout(timer);
    }
  }

  /** Optimise a prompt. Returns the full report. */
  optimizePrompt(prompt, opts = {}) {
    return this._post("/optimize", {
      prompt,
      system: opts.system,
      level: opts.level || "balanced",
      security: opts.security || "warn",
      priority: opts.priority || "balanced",
      providers: opts.providers,
      select_model: opts.selectModel !== false,
    });
  }

  /** Optimise a model answer. */
  optimizeOutput(text, opts = {}) {
    return this._post("/optimize/output", { text, level: opts.level || "balanced" });
  }

  async models() {
    const res = await fetch(this.baseUrl + "/models");
    return res.json();
  }

  async health() {
    const res = await fetch(this.baseUrl + "/health");
    return res.json();
  }

  /**
   * Convenience wrapper. Optimise, call your own LLM function, then optimise
   * the answer. `callModel(prompt, model)` must return a string.
   */
  async wrap(prompt, callModel, opts = {}) {
    const pre = await this.optimizePrompt(prompt, opts);
    const raw = await callModel(pre.optimized_prompt, pre.model_choice?.model);
    const post = await this.optimizeOutput(raw, opts);
    return {
      answer: post.text,
      rawAnswer: raw,
      tokensSaved: (pre.tokens_saved || 0) + (post.tokens_saved || 0),
      report: { prompt: pre, output: post },
    };
  }
}

export default GenAIOptimizer;
