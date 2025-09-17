import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
import os
import json
import logging
import sys
import queue
from logging import Handler

# Import the new AI Manager
from ai_module import AIManager

# --- Setup for Queue-Based Logging ---
class QueueHandler(Handler):
    """A custom logging handler that puts logs into a queue for the GUI."""
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

# --- Initial Dependency Check ---
try:
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import torch
    import requests # Added for ai_module
except ImportError:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Missing Dependencies",
        "Required libraries are missing. Please run:\n\n"
        "pip install langchain langchain_community sentence-transformers "
        "PyMuPDF Pillow torch faiss-cpu requests huggingface_hub\n\n"
        "NOTE: For GPU support, ensure PyTorch is installed with CUDA. "
        "See the official PyTorch website for installation instructions."
    )
    exit()

# --- Global Constants ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
SETTINGS_FILE = "settings.json"
FAISS_INDEX_PATH = "faiss_index"

class RagApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Multi-AI RAG Assistant")
        self.root.geometry("900x850")

        self.pdf_files = []
        self.vector_store = None
        self.processing_thread = None
        self.settings = {}

        # Instantiate the AI Manager
        self.ai_manager = AIManager(self)

        self.create_widgets()
        self.configure_logging()
        self.load_settings()
        
        self.load_existing_db()

    def configure_logging(self):
        """Configures queue-based logging for thread-safe GUI updates."""
        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.queue_handler.setFormatter(formatter)
        
        self.root_logger = logging.getLogger()
        self.root_logger.addHandler(self.queue_handler)
        self.root_logger.setLevel(logging.INFO) 
        
        self.root.after(100, self.poll_log_queue)

    def poll_log_queue(self):
        """Periodically checks the log queue and updates the GUI."""
        try:
            while True:
                record = self.log_queue.get(block=False)
                self.display_log_message(record)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.poll_log_queue)

    def create_widgets(self):
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Top section for Prompt and Output ---
        top_io_frame = tk.Frame(main_frame)
        top_io_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        io_paned_window = ttk.PanedWindow(top_io_frame, orient=tk.HORIZONTAL)
        io_paned_window.pack(fill=tk.BOTH, expand=True)

        prompt_frame = ttk.LabelFrame(io_paned_window, text="User Prompt", padding=5)
        self.prompt_input = scrolledtext.ScrolledText(prompt_frame, wrap=tk.WORD, height=10)
        self.prompt_input.pack(fill=tk.BOTH, expand=True)
        io_paned_window.add(prompt_frame, weight=1)

        output_frame = ttk.LabelFrame(io_paned_window, text="AI Output", padding=5)
        self.ai_output = scrolledtext.ScrolledText(output_frame, state='disabled', wrap=tk.WORD, height=10)
        self.ai_output.pack(fill=tk.BOTH, expand=True)
        io_paned_window.add(output_frame, weight=1)

        # --- Configuration Section with Tabs ---
        config_notebook = ttk.Notebook(main_frame)
        config_notebook.pack(fill=tk.X, pady=5)

        for provider in self.ai_manager.AI_PROVIDERS:
            tab = ttk.Frame(config_notebook, padding=10)
            config_notebook.add(tab, text=provider)
            self.ai_manager.create_provider_tab(tab, provider)

        # --- Middle section for App settings and PDF upload ---
        middle_frame = tk.Frame(main_frame)
        middle_frame.pack(fill=tk.X, pady=5)
        middle_frame.columnconfigure(0, weight=1)
        middle_frame.columnconfigure(1, weight=1)

        pdf_frame = ttk.LabelFrame(middle_frame, text="Upload New PDFs to Knowledge Base", padding=10)
        pdf_frame.grid(row=0, column=0, sticky="nsew", padx=(0,5))
        self.load_pdf_button = tk.Button(pdf_frame, text="Load PDFs", command=self.load_pdfs)
        self.load_pdf_button.pack(side=tk.LEFT)
        self.pdf_listbox = tk.Listbox(pdf_frame, height=3)
        self.pdf_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        self.process_pdfs_button = tk.Button(pdf_frame, text="Create/Update DB", command=self.start_processing_pdfs, state=tk.DISABLED)
        self.process_pdfs_button.pack(side=tk.LEFT, padx=(10, 0))

        app_settings_frame = ttk.LabelFrame(middle_frame, text="App Settings", padding=10)
        app_settings_frame.grid(row=0, column=1, sticky="nsew", padx=(5,0))
        tk.Label(app_settings_frame, text="Logging Level:").pack(side=tk.LEFT, padx=(0, 10))
        self.log_level_var = tk.StringVar()
        self.log_level_menu = ttk.Combobox(
            app_settings_frame, textvariable=self.log_level_var,
            values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], state="readonly"
        )
        self.log_level_menu.pack(side=tk.LEFT)
        self.log_level_menu.bind("<<ComboboxSelected>>", self.on_log_level_change)
        
        indexed_frame = ttk.LabelFrame(main_frame, text="Documents in Knowledge Base", padding=10)
        indexed_frame.pack(fill=tk.X, pady=10)
        self.indexed_files_listbox = tk.Listbox(indexed_frame, height=4)
        self.indexed_files_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        refresh_button = tk.Button(indexed_frame, text="Refresh List", command=self.display_indexed_documents)
        refresh_button.pack(side=tk.LEFT, padx=(10, 0))

        self.status_label = tk.Label(main_frame, text="Status: Ready.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X, pady=5)

        # --- Bottom section for Logs ---
        log_frame = ttk.LabelFrame(main_frame, text="Logs", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self.console_output = scrolledtext.ScrolledText(log_frame, state='disabled', wrap=tk.WORD, height=10)
        self.console_output.pack(fill=tk.BOTH, expand=True)

    def on_log_level_change(self, event=None):
        """Updates the logger's level and saves settings."""
        level = self.log_level_var.get()
        if level:
            self.root_logger.setLevel(level)
            logging.info(f"Logging level set to {level}")
            self.save_settings()

    def display_indexed_documents(self):
        """Displays source files from the FAISS docstore."""
        self.indexed_files_listbox.delete(0, tk.END)
        if not self.vector_store or not hasattr(self.vector_store, 'docstore'):
            self.indexed_files_listbox.insert(tk.END, "No knowledge base loaded.")
            return
        try:
            docstore_docs = self.vector_store.docstore._dict.values()
            source_files = sorted(list({doc.metadata.get('source', 'Unknown Source') for doc in docstore_docs}))
            for f in source_files or ["No documents found in index."]:
                self.indexed_files_listbox.insert(tk.END, os.path.basename(f))
        except Exception as e:
            logging.error(f"Failed to read from FAISS docstore: {e}")
            self.indexed_files_listbox.insert(tk.END, "Error reading index.")

    def load_settings(self):
        """Loads settings from file or creates a default file."""
        if not os.path.exists(SETTINGS_FILE):
            logging.info("Settings file not found. Creating with defaults.")
            self.settings["ai_settings"] = self.ai_manager.get_default_settings()
            self.settings["log_level"] = "INFO"
            self.save_settings()
        
        try:
            with open(SETTINGS_FILE, 'r') as f:
                self.settings = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Could not load settings file: {e}. Loading defaults.")
            self.settings["ai_settings"] = self.ai_manager.get_default_settings()
            self.settings["log_level"] = "INFO"

        log_level = self.settings.get("log_level", "INFO")
        self.log_level_var.set(log_level)
        self.root_logger.setLevel(log_level)
        self.ai_manager.load_provider_settings(self.settings.get("ai_settings", {}))

    def save_settings(self, event=None):
        """Saves current UI settings to the settings file."""
        self.settings["log_level"] = self.log_level_var.get()
        self.settings["ai_settings"] = self.ai_manager.get_settings_from_ui()
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except IOError as e:
            logging.error(f"Could not save settings: {e}")

    def get_embedding_model(self):
        """Initializes and returns the embedding model."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {device} for embedding model.")
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        model_path = os.path.join(sys._MEIPASS, 'all-MiniLM-L6-v2') if getattr(sys, 'frozen', False) else model_name
        return HuggingFaceEmbeddings(model_name=model_path, model_kwargs={'device': device})

    def load_existing_db(self):
        """Loads a persisted FAISS index from disk."""
        if os.path.exists(FAISS_INDEX_PATH):
            try:
                self.status_label.config(text="Status: Loading knowledge base...")
                self.root.update_idletasks()
                embedding_model = self.get_embedding_model()
                self.vector_store = FAISS.load_local(
                    folder_path=FAISS_INDEX_PATH,
                    embeddings=embedding_model,
                    allow_dangerous_deserialization=True
                )
                self.on_processing_complete()
                logging.info("Existing knowledge base loaded successfully.")
            except Exception as e:
                logging.exception("Failed to load existing knowledge base.")
                messagebox.showerror("DB Load Error", f"Could not load index. Error: {e}")

    def load_pdfs(self):
        filepaths = filedialog.askopenfilenames(title="Select PDF files", filetypes=(("PDF files", "*.pdf"),))
        if filepaths:
            self.pdf_files.extend(filepaths)
            self.update_pdf_listbox()
            self.process_pdfs_button.config(state=tk.NORMAL)

    def update_pdf_listbox(self):
        self.pdf_listbox.delete(0, tk.END)
        for f in self.pdf_files:
            self.pdf_listbox.insert(tk.END, os.path.basename(f))
            
    def start_processing_pdfs(self):
        if not self.pdf_files: return
        if os.path.exists(FAISS_INDEX_PATH) and not messagebox.askyesno("Confirm", "Overwrite existing knowledge base?"):
            return
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Busy", "Already processing. Please wait.")
            return
        
        self.processing_thread = threading.Thread(target=self.process_pdfs_backend, daemon=True)
        self.processing_thread.start()
        self.load_pdf_button.config(state=tk.DISABLED)
        self.process_pdfs_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Creating knowledge base...")
        
    def process_pdfs_backend(self):
        try:
            logging.info("Starting to create new knowledge base...")
            all_docs = [doc for pdf_path in self.pdf_files for doc in PyMuPDFLoader(pdf_path).load()]
            texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(all_docs)
            logging.info(f"Text split into {len(texts)} chunks.")
            embedding_model = self.get_embedding_model()
            self.vector_store = FAISS.from_documents(texts, embedding_model)
            self.vector_store.save_local(FAISS_INDEX_PATH)
            logging.info("Knowledge base created and saved successfully.")
            self.root.after(0, self.on_processing_complete)
        except Exception as e:
            logging.exception("An error occurred during PDF processing.")
            self.root.after(0, lambda: self.on_processing_error(e))

    def on_processing_complete(self):
        self.status_label.config(text="Status: Ready.")
        self.load_pdf_button.config(state=tk.NORMAL)
        self.process_pdfs_button.config(state=tk.DISABLED)
        self.pdf_files.clear()
        self.update_pdf_listbox()
        self.display_indexed_documents()

    def on_processing_error(self, error):
        messagebox.showerror("Processing Error", f"An error occurred: {error}\nCheck logs.")
        self.status_label.config(text="Status: Error during processing.")
        self.load_pdf_button.config(state=tk.NORMAL)
        self.process_pdfs_button.config(state=tk.NORMAL)

    def start_ai_processing(self):
        """Starts the AI processing in a separate thread."""
        self.status_label.config(text="Status: Thinking...")
        threading.Thread(target=self.ai_manager.process_ai_request, daemon=True).start()
    
    def display_llm_response(self, response):
        """Callback to display the AI's response in the GUI."""
        self.ai_output.config(state='normal')
        self.ai_output.delete('1.0', tk.END)
        self.ai_output.insert('1.0', response)
        self.ai_output.config(state='disabled')
        self.status_label.config(text="Status: Ready.")

    def display_log_message(self, message):
        self.console_output.config(state='normal')
        self.console_output.insert(tk.END, message + "\n")
        self.console_output.config(state='disabled')
        self.console_output.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = RagApp(root)
    root.mainloop()

