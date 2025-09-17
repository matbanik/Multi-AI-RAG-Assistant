import tkinter as tk
from tkinter import ttk, font, scrolledtext, messagebox
import webbrowser
import requests
import threading
import time
import random
import json
import logging

try:
    from huggingface_hub import InferenceClient
    from huggingface_hub.utils import HfHubHTTPError
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Default settings derived from settings.txt
DEFAULT_AI_SETTINGS = {
    "Google AI": {"model": "gemini-1.5-pro-latest", "MODELS_LIST": ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-1.0-pro"], "temperature": "0.7", "topK": "40", "topP": "0.95", "maxOutputTokens": "8192"},
    "Anthropic AI": {"model": "claude-3-5-sonnet-20240620", "MODELS_LIST": ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"], "system": "You are a helpful assistant.", "max_tokens": "4096", "temperature": "0.7", "top_p": "0.9", "top_k": "40"},
    "OpenAI": {"model": "gpt-4o", "MODELS_LIST": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o-mini"], "system_prompt": "You are a helpful assistant.", "temperature": "0.7", "max_tokens": "4096", "top_p": "1.0", "frequency_penalty": "0.0", "presence_penalty": "0.0", "seed": ""},
    "Cohere AI": {"model": "command-r-plus", "MODELS_LIST": ["command-r-plus", "command-r", "command"], "preamble": "You are a helpful assistant.", "temperature": "0.7", "max_tokens": "4000", "k": "50", "p": "0.75", "frequency_penalty": "0.0", "presence_penalty": "0.0"},
    "HuggingFace AI": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "MODELS_LIST": ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2", "google/gemma-7b-it"], "system_prompt": "You are a helpful assistant.", "max_tokens": "4096", "temperature": "0.7", "top_p": "0.95", "seed": ""},
    "Groq AI": {"model": "llama3-70b-8192", "MODELS_LIST": ["llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"], "system_prompt": "You are a helpful assistant.", "temperature": "0.7", "max_tokens": "8192", "top_p": "1.0", "seed": ""},
    "OpenRouterAI": {"model": "anthropic/claude-3.5-sonnet", "MODELS_LIST": ["anthropic/claude-3.5-sonnet", "google/gemini-flash-1.5", "meta-llama/llama-3-8b-instruct", "openai/gpt-4o-mini"], "system_prompt": "You are a helpful assistant.", "temperature": "0.7", "max_tokens": "4096", "top_p": "1.0", "top_k": "0", "frequency_penalty": "0.0", "presence_penalty": "0.0"}
}

PROVIDER_PARAMS = {
    "Google AI": {"system_prompt": False, "params": {"temperature": {"type": float, "tip": "0.0-2.0"}, "topK": {"type": int, "tip": "1-100"}, "topP": {"type": float, "tip": "0.0-1.0"}, "maxOutputTokens": {"type": int, "tip": "1-8192"}}},
    "Cohere AI": {"system_prompt": True, "system_prompt_key": "preamble", "params": {"temperature": {"type": float, "tip": "0.0-1.0"}, "max_tokens": {"type": int, "tip": "1-4000"}, "k": {"type": int, "tip": "0-500"}, "p": {"type": float, "tip": "0.0-1.0"}, "frequency_penalty": {"type": float, "tip": "0.0-1.0"}, "presence_penalty": {"type": float, "tip": "0.0-1.0"}}},
    "HuggingFace AI": {"system_prompt": True, "params": {"max_tokens": {"type": int, "tip": "e.g., 4096"}, "temperature": {"type": float, "tip": "0.0-2.0"}, "top_p": {"type": float, "tip": "0.0-1.0"}, "seed": {"type": int, "tip": "Integer"}}},
    "Groq AI": {"system_prompt": True, "params": {"temperature": {"type": float, "tip": "0.0-2.0"}, "max_tokens": {"type": int, "tip": "e.g., 8192"}, "top_p": {"type": float, "tip": "0.0-1.0"}, "seed": {"type": int, "tip": "Integer"}}},
    "OpenRouterAI": {"system_prompt": True, "params": {"temperature": {"type": float, "tip": "0.0-2.0"}, "max_tokens": {"type": int, "tip": "e.g., 4096"}, "top_p": {"type": float, "tip": "0.0-1.0"}, "top_k": {"type": int, "tip": "0-100"}, "frequency_penalty": {"type": float, "tip": "-2.0-2.0"}, "presence_penalty": {"type": float, "tip": "-2.0-2.0"}}},
    "Anthropic AI": {"system_prompt": True, "system_prompt_key": "system", "params": {"max_tokens": {"type": int, "tip": "1-8192"}, "temperature": {"type": float, "tip": "0.0-1.0"}, "top_p": {"type": float, "tip": "0.0-1.0"}, "top_k": {"type": int, "tip": "1-100"}}},
    "OpenAI": {"system_prompt": True, "params": {"temperature": {"type": float, "tip": "0.0-2.0"}, "max_tokens": {"type": int, "tip": "e.g., 4096"}, "top_p": {"type": float, "tip": "0.0-1.0"}, "frequency_penalty": {"type": float, "tip": "-2.0-2.0"}, "presence_penalty": {"type": float, "tip": "-2.0-2.0"}, "seed": {"type": int, "tip": "Integer"}}},
}

class ToolTip:
    def __init__(self, widget, text):
        self.widget, self.text, self.tooltip = widget, text, None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)
    def show_tip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, justify='left', background="#ffffe0", relief='solid', borderwidth=1)
        label.pack(ipadx=1)
    def hide_tip(self, event=None):
        if self.tooltip: self.tooltip.destroy()
        self.tooltip = None

class AIManager:
    def __init__(self, app_instance):
        self.app = app_instance
        self.provider_widgets = {}
        self.ai_provider_urls = {"Google AI": "https://aistudio.google.com/apikey", "Cohere AI": "https://dashboard.cohere.com/api-keys", "HuggingFace AI": "https://huggingface.co/settings/tokens", "Groq AI": "https://console.groq.com/keys", "OpenRouterAI": "https://openrouter.ai/settings/keys", "Anthropic AI": "https://console.anthropic.com/settings/keys", "OpenAI": "https://platform.openai.com/settings/organization/api-keys"}
        self.MAX_RETRIES, self.BASE_DELAY = 3, 1
        self.AI_PROVIDERS = list(DEFAULT_AI_SETTINGS.keys())

    def get_default_settings(self): return DEFAULT_AI_SETTINGS

    def create_provider_tab(self, parent, provider_name):
        widgets, provider_config = {}, PROVIDER_PARAMS.get(provider_name, {})
        top_frame = tk.Frame(parent); top_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(top_frame, text="API Key:").pack(side=tk.LEFT, padx=(0, 5))
        api_key_var = tk.StringVar(); widgets['api_key_var'] = api_key_var
        ttk.Entry(top_frame, textvariable=api_key_var, width=25, show="*").pack(side=tk.LEFT)
        api_key_var.trace_add("write", self.app.save_settings)
        if url := self.ai_provider_urls.get(provider_name):
            link = ttk.Label(top_frame, text="Get Key", foreground="blue", cursor="hand2", font=font.Font(underline=True))
            link.pack(side=tk.LEFT, padx=5); link.bind("<Button-1>", lambda e, u=url: webbrowser.open_new(u))
        tk.Label(top_frame, text="Model:").pack(side=tk.LEFT, padx=(10, 5))
        model_var = tk.StringVar(); widgets['model_var'] = model_var
        model_menu = ttk.Combobox(top_frame, textvariable=model_var, state="readonly", width=30); widgets['model_menu'] = model_menu
        model_menu.pack(side=tk.LEFT); model_var.trace_add("write", self.app.save_settings)
        ttk.Button(top_frame, text="\u270E", width=3, command=lambda: self.open_model_editor(provider_name)).pack(side=tk.LEFT, padx=(2,10))
        ttk.Button(top_frame, text="Process", command=self.app.start_ai_processing).pack(side=tk.RIGHT, padx=5)
        
        content_frame = tk.Frame(parent); content_frame.pack(fill=tk.BOTH, expand=True)
        content_frame.columnconfigure(1, weight=1)
        if provider_config.get("system_prompt"):
            system_frame = ttk.LabelFrame(content_frame, text="System Prompt")
            system_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 5))
            system_prompt_text = scrolledtext.ScrolledText(system_frame, height=8, width=35, wrap=tk.WORD); widgets['system_prompt_text'] = system_prompt_text
            system_prompt_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            system_prompt_text.bind("<KeyRelease>", self.app.save_settings)

        params_frame = ttk.LabelFrame(content_frame, text="Parameters"); params_frame.grid(row=0, column=1, sticky="nsew")
        canvas = tk.Canvas(params_frame); scrollbar = ttk.Scrollbar(params_frame, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas); scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw"); canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")
        for i, (p_name, p_config) in enumerate(provider_config.get("params", {}).items()):
            ttk.Label(scrollable, text=f"{p_name}:").grid(row=i, column=0, sticky="w", padx=5, pady=2)
            param_var = tk.StringVar(); widgets[f'param_{p_name}_var'] = param_var
            entry = ttk.Entry(scrollable, textvariable=param_var, width=15)
            entry.grid(row=i, column=1, sticky="ew", padx=5, pady=2); param_var.trace_add("write", self.app.save_settings)
            ToolTip(entry, p_config.get("tip", ""))
        scrollable.columnconfigure(1, weight=1); self.provider_widgets[provider_name] = widgets

    def open_model_editor(self, provider_name):
        dialog = tk.Toplevel(self.app.root); dialog.title(f"Edit {provider_name} Models"); dialog.geometry("350x250"); dialog.transient(self.app.root); dialog.grab_set()
        x = self.app.root.winfo_x() + (self.app.root.winfo_width() // 2) - 175
        y = self.app.root.winfo_y() + (self.app.root.winfo_height() // 2) - 125
        dialog.geometry(f"+{x}+{y}")
        ttk.Label(dialog, text="One model per line. First line is default.").pack(pady=5)
        text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, height=8); text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        provider_settings = self.app.settings.get("ai_settings", {}).get(provider_name, {})
        text.insert("1.0", "\n".join(provider_settings.get("MODELS_LIST", [])))
        ttk.Button(dialog, text="Save", command=lambda: self.save_model_list(provider_name, text, dialog)).pack(pady=5)

    def save_model_list(self, provider_name, text_area, dialog):
        new_list = [line.strip() for line in text_area.get("1.0", tk.END).strip().splitlines() if line.strip()]
        if not new_list: return messagebox.showwarning("Empty List", "Model list cannot be empty.", parent=dialog)
        self.app.settings["ai_settings"][provider_name]["MODELS_LIST"] = new_list
        self.app.settings["ai_settings"][provider_name]["model"] = new_list[0]
        widgets = self.provider_widgets[provider_name]
        widgets['model_var'].set(new_list[0]); widgets['model_menu']['values'] = new_list
        self.app.save_settings(); dialog.destroy()

    def load_provider_settings(self, ai_settings):
        for name, widgets in self.provider_widgets.items():
            defaults = DEFAULT_AI_SETTINGS.get(name, {})
            saved = ai_settings.get(name, {})
            get = lambda key, fb="": saved.get(key, defaults.get(key, fb))
            widgets['api_key_var'].set(saved.get("api_key", ""))
            model_list = get("MODELS_LIST", [])
            widgets['model_menu']['values'] = model_list
            widgets['model_var'].set(get("model", model_list[0] if model_list else ""))
            if 'system_prompt_text' in widgets:
                sp_key = PROVIDER_PARAMS.get(name, {}).get("system_prompt_key", "system_prompt")
                prompt = saved.get(sp_key, defaults.get(sp_key, ""))
                widgets['system_prompt_text'].delete("1.0", tk.END); widgets['system_prompt_text'].insert("1.0", prompt)
            for p_name in PROVIDER_PARAMS.get(name, {}).get("params", {}):
                widgets[f'param_{p_name}_var'].set(get(p_name))

    def get_settings_from_ui(self):
        all_settings = {}
        for name, widgets in self.provider_widgets.items():
            s = {"api_key": widgets['api_key_var'].get(), "model": widgets['model_var'].get(), "MODELS_LIST": list(widgets['model_menu']['values'])}
            if 'system_prompt_text' in widgets:
                key = PROVIDER_PARAMS.get(name, {}).get("system_prompt_key", "system_prompt")
                s[key] = widgets['system_prompt_text'].get("1.0", tk.END).strip()
            for p_name in PROVIDER_PARAMS.get(name, {}).get("params", {}):
                s[p_name] = widgets[f'param_{p_name}_var'].get()
            all_settings[name] = s
        return all_settings
    
    def process_ai_request(self):
        tab_index = self.app.root.nametowidget('.!frame.!notebook').index('current')
        provider = self.app.root.nametowidget('.!frame.!notebook').tab(tab_index, "text")
        s = self.app.settings.get("ai_settings", {}).get(provider, {})
        user_prompt = self.app.prompt_input.get("1.0", tk.END).strip()
        if not s.get("api_key"): return self.app.root.after(0, self.app.display_llm_response, f"Error: API Key for {provider} is missing.")
        if not user_prompt: return self.app.root.after(0, self.app.display_llm_response, "Error: User Prompt cannot be empty.")
        
        config = PROVIDER_PARAMS.get(provider, {}); sp_key = config.get("system_prompt_key", "system_prompt")
        system_prompt = s.get(sp_key, "")
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt and provider not in ["Cohere AI", "Anthropic AI", "Google AI"] else []) + [{"role": "user", "content": user_prompt}]
        
        if provider == "HuggingFace AI": return self._handle_huggingface(s, messages, config)

        headers, payload = {'Content-Type': 'application/json'}, {}
        if provider == "Google AI":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{s.get('model')}:generateContent?key={s.get('api_key')}"
            payload = {"contents": [{"parts": [{"text": user_prompt}]}], "generationConfig": {}}
            params_target = payload["generationConfig"]
        else:
            if provider == "Cohere AI": url, payload['message'] = "https://api.cohere.com/v1/chat", user_prompt
            elif provider == "Anthropic AI": url, payload['messages'] = "https://api.anthropic.com/v1/messages", messages
            else: url, payload['messages'] = {"OpenAI": "https://api.openai.com/v1/chat/completions", "Groq AI": "https://api.groq.com/openai/v1/chat/completions", "OpenRouterAI": "https://openrouter.ai/api/v1/chat/completions"}[provider], messages
            headers['Authorization'] = f"Bearer {s.get('api_key')}"
            if provider == "Anthropic AI": headers.update({'x-api-key': s.get('api_key'), 'anthropic-version': '2023-06-01'})
            payload.update({"model": s.get('model')})
            if system_prompt and sp_key in s: payload[sp_key] = system_prompt
            params_target = payload

        for name, p_conf in config.get("params", {}).items():
            if val_str := s.get(name):
                try: params_target[name] = p_conf["type"](val_str)
                except (ValueError, TypeError): logging.warning(f"Skipping invalid param '{name}': '{val_str}'")
        
        logging.debug(f"Sending to {provider}: {json.dumps(payload, indent=2)}")
        self._make_request(provider, url, headers, payload)

    def _handle_huggingface(self, s, messages, config):
        if not HUGGINGFACE_AVAILABLE: return self.app.root.after(0, self.app.display_llm_response, "Error: huggingface_hub missing.")
        try:
            params = {}
            for name, p_conf in config.get("params", {}).items():
                if val_str := s.get(name):
                    try: params[name] = p_conf["type"](val_str)
                    except (ValueError, TypeError): pass
            logging.debug(f"Sending to HuggingFace: model={s.get('model')}, messages={messages}, params={params}")
            resp = InferenceClient(token=s.get('api_key')).chat_completion(messages=messages, model=s.get('model'), **params)
            logging.debug(f"Received from HuggingFace: {resp}")
            self.app.root.after(0, self.app.display_llm_response, resp.choices[0].message.content)
        except HfHubHTTPError as e:
            self.app.root.after(0, self.app.display_llm_response, f"HF API Error: {e.response.status_code}\n{e.response.text}")
        except Exception as e:
            self.app.root.after(0, self.app.display_llm_response, f"HF Client Error: {e}")

    def _make_request(self, provider, url, headers, payload):
        for i in range(self.MAX_RETRIES):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=60)
                logging.debug(f"Received from {provider}: {resp.text}")
                resp.raise_for_status()
                data = resp.json()
                text = {"Google AI": lambda d: d['candidates'][0]['content']['parts'][0]['text'], "Cohere AI": lambda d: d['text'], "Anthropic AI": lambda d: d['content'][0]['text']}.get(provider, lambda d: d['choices'][0]['message']['content'])(data)
                return self.app.root.after(0, self.app.display_llm_response, text)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429 and i < self.MAX_RETRIES - 1:
                    delay = self.BASE_DELAY * (2 ** i) + random.uniform(0, 1)
                    logging.warning(f"Rate limit. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else: return self.app.root.after(0, self.app.display_llm_response, f"API Error: {e}\n{e.response.text}")
            except Exception as e: return self.app.root.after(0, self.app.display_llm_response, f"Request Error: {e}")
        self.app.root.after(0, self.app.display_llm_response, "Error: Max retries exceeded.")

