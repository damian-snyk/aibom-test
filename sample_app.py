"""
Comprehensive AI/ML Application Sample
Demonstrates usage of various AI/ML packages for AIBOM testing.

Includes:
- OpenAI (GPT-4, GPT-3.5, Embeddings, DALL-E, Whisper)
- Anthropic (Claude 3, Claude 3.5)
- AWS Bedrock (Claude, Titan, Llama)
- LangChain & LlamaIndex
- PyTorch & TensorFlow
- Hugging Face Transformers
- spaCy NLP
- Scikit-learn
- NumPy, Pandas, Seaborn
"""

import os
import json
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime

# Environment configuration
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# Data Science & Numerical Computing
# =============================================================================
import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.spatial.distance import cosine

# =============================================================================
# Data Visualization
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# AWS SDK
# =============================================================================
import boto3
from botocore.config import Config

# =============================================================================
# OpenAI SDK
# =============================================================================
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types import CreateEmbeddingResponse

# =============================================================================
# Anthropic SDK
# =============================================================================
import anthropic
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message, ContentBlock

# =============================================================================
# PyTorch Imports
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torchvision
from torchvision import transforms, models as torchvision_models
from torchvision.datasets import MNIST, CIFAR10, ImageNet
import torchaudio
from torchaudio import transforms as audio_transforms

# =============================================================================
# TensorFlow Imports
# =============================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models as keras_models
from tensorflow.keras.applications import (
    ResNet50, VGG16, InceptionV3, MobileNetV2, EfficientNetB0
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub

# =============================================================================
# Hugging Face Ecosystem
# =============================================================================
from transformers import (
    # Tokenizers
    AutoTokenizer,
    BertTokenizer,
    GPT2Tokenizer,
    RobertaTokenizer,
    T5Tokenizer,
    LlamaTokenizer,
    # Models - General
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    # Specific Models
    BertModel,
    BertForSequenceClassification,
    GPT2LMHeadModel,
    GPT2Model,
    RobertaModel,
    T5ForConditionalGeneration,
    LlamaForCausalLM,
    MistralForCausalLM,
    Phi3ForCausalLM,
    GemmaForCausalLM,
    Qwen2ForCausalLM,
    FalconForCausalLM,
    # Vision Models
    ViTModel,
    ViTForImageClassification,
    CLIPModel,
    CLIPProcessor,
    # Pipelines
    pipeline,
    # Training
    Trainer,
    TrainingArguments,
    # Utilities
    GenerationConfig,
)
from datasets import load_dataset, Dataset as HFDataset
from accelerate import Accelerator
from sentence_transformers import SentenceTransformer

# =============================================================================
# LlamaIndex
# =============================================================================
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document as LlamaDocument,
    Settings,
    StorageContext,
    ServiceContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.llms.anthropic import Anthropic as LlamaAnthropic
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# =============================================================================
# spaCy NLP
# =============================================================================
import spacy
from spacy.tokens import Doc, Token, Span
from spacy.matcher import Matcher, PhraseMatcher
from spacy.language import Language

# =============================================================================
# Scikit-learn
# =============================================================================
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    RandomForestRegressor,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    mean_squared_error
)
from sklearn.pipeline import Pipeline

# =============================================================================
# LangChain
# =============================================================================
from langchain.chains import ConversationChain, RetrievalQA, LLMChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document, HumanMessage, AIMessage, SystemMessage
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch, Chroma, FAISS
from langchain_community.document_loaders import TextLoader, PDFLoader
from langchain_community.llms import OpenAI as LangChainOpenAI
from langchain_community.chat_models import ChatOpenAI, ChatAnthropic

# =============================================================================
# Vector Databases
# =============================================================================
from opensearchpy import OpenSearch, RequestsHttpConnection
import chromadb
from chromadb.config import Settings as ChromaSettings

# =============================================================================
# Data validation
# =============================================================================
from pydantic import BaseModel, Field

# =============================================================================
# Utilities
# =============================================================================
import requests
import aiohttp
import asyncio
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential


# =============================================================================
# Pydantic Models for structured data
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for AI queries."""
    question: str = Field(..., description="The user's question")
    context: Optional[str] = Field(None, description="Additional context")
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    """Response model for AI queries."""
    answer: str
    tokens_used: int
    model_id: str
    sources: List[str] = []


class EmbeddingRequest(BaseModel):
    """Request for generating embeddings."""
    texts: List[str]
    model: str = "text-embedding-3-small"


# =============================================================================
# OpenAI Client
# =============================================================================

class OpenAIClient:
    """Client for OpenAI API interactions."""
    
    # Model identifiers
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4_TURBO_PREVIEW = "gpt-4-turbo-preview"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_35_TURBO_16K = "gpt-3.5-turbo-16k"
    
    # Embedding models
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    
    # Image models
    DALL_E_3 = "dall-e-3"
    DALL_E_2 = "dall-e-2"
    
    # Audio models
    WHISPER_1 = "whisper-1"
    TTS_1 = "tts-1"
    TTS_1_HD = "tts-1-hd"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        self.token_counter = tiktoken.encoding_for_model("gpt-4")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.token_counter.encode(text))
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> ChatCompletion:
        """Create a chat completion."""
        model = model or self.GPT_4O
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response
    
    def create_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """Create embeddings for texts."""
        model = model or self.TEXT_EMBEDDING_3_SMALL
        response = self.client.embeddings.create(
            model=model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def generate_image(
        self,
        prompt: str,
        model: str = None,
        size: str = "1024x1024",
        quality: str = "standard"
    ) -> str:
        """Generate an image using DALL-E."""
        model = model or self.DALL_E_3
        response = self.client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=1
        )
        return response.data[0].url
    
    def transcribe_audio(self, audio_file_path: str) -> str:
        """Transcribe audio using Whisper."""
        with open(audio_file_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model=self.WHISPER_1,
                file=audio_file
            )
        return response.text
    
    async def async_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = None
    ) -> ChatCompletion:
        """Async chat completion."""
        model = model or self.GPT_4O
        response = await self.async_client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response


# =============================================================================
# Anthropic Client
# =============================================================================

class AnthropicClient:
    """Client for Anthropic API interactions."""
    
    # Claude 3 models
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    
    # Claude 3.5 models
    CLAUDE_35_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_35_HAIKU = "claude-3-5-haiku-20241022"
    
    # Legacy models
    CLAUDE_2 = "claude-2.1"
    CLAUDE_INSTANT = "claude-instant-1.2"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=self.api_key)
        self.async_client = AsyncAnthropic(api_key=self.api_key)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def create_message(
        self,
        prompt: str,
        model: str = None,
        system: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> Message:
        """Create a message using Claude."""
        model = model or self.CLAUDE_35_SONNET
        
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system:
            kwargs["system"] = system
        
        return self.client.messages.create(**kwargs)
    
    def create_with_tools(
        self,
        prompt: str,
        tools: List[Dict],
        model: str = None
    ) -> Message:
        """Create a message with tool use."""
        model = model or self.CLAUDE_35_SONNET
        return self.client.messages.create(
            model=model,
            max_tokens=1024,
            tools=tools,
            messages=[{"role": "user", "content": prompt}]
        )
    
    def create_with_vision(
        self,
        prompt: str,
        image_data: str,
        media_type: str = "image/jpeg",
        model: str = None
    ) -> Message:
        """Create a message with image input."""
        model = model or self.CLAUDE_35_SONNET
        return self.client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )
    
    async def async_create_message(
        self,
        prompt: str,
        model: str = None
    ) -> Message:
        """Async message creation."""
        model = model or self.CLAUDE_35_SONNET
        return await self.async_client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )


# =============================================================================
# AWS Bedrock Client
# =============================================================================

class BedrockClient:
    """Client for AWS Bedrock interactions."""
    
    # Anthropic Claude models on Bedrock
    CLAUDE_3_OPUS_BEDROCK = "anthropic.claude-3-opus-20240229-v1:0"
    CLAUDE_3_SONNET_BEDROCK = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_3_HAIKU_BEDROCK = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_35_SONNET_BEDROCK = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    
    # Amazon Titan models
    TITAN_TEXT_EXPRESS = "amazon.titan-text-express-v1"
    TITAN_TEXT_LITE = "amazon.titan-text-lite-v1"
    TITAN_EMBED_TEXT = "amazon.titan-embed-text-v1"
    TITAN_EMBED_TEXT_V2 = "amazon.titan-embed-text-v2:0"
    TITAN_EMBED_IMAGE = "amazon.titan-embed-image-v1"
    
    # Meta Llama models
    LLAMA_3_8B = "meta.llama3-8b-instruct-v1:0"
    LLAMA_3_70B = "meta.llama3-70b-instruct-v1:0"
    LLAMA_31_8B = "meta.llama3-1-8b-instruct-v1:0"
    LLAMA_31_70B = "meta.llama3-1-70b-instruct-v1:0"
    LLAMA_31_405B = "meta.llama3-1-405b-instruct-v1:0"
    
    # Mistral models
    MISTRAL_7B = "mistral.mistral-7b-instruct-v0:2"
    MISTRAL_LARGE = "mistral.mistral-large-2402-v1:0"
    MIXTRAL_8X7B = "mistral.mixtral-8x7b-instruct-v0:1"
    
    # Cohere models
    COHERE_COMMAND = "cohere.command-text-v14"
    COHERE_COMMAND_LIGHT = "cohere.command-light-text-v14"
    COHERE_EMBED_ENGLISH = "cohere.embed-english-v3"
    COHERE_EMBED_MULTILINGUAL = "cohere.embed-multilingual-v3"
    
    # AI21 Labs models
    AI21_JURASSIC_ULTRA = "ai21.j2-ultra-v1"
    AI21_JURASSIC_MID = "ai21.j2-mid-v1"
    
    # Stability AI models
    STABLE_DIFFUSION_XL = "stability.stable-diffusion-xl-v1"
    
    def __init__(self, region_name: str = "us-east-1"):
        self.region_name = region_name
        self.config = Config(
            region_name=region_name,
            retries={"max_attempts": 3, "mode": "adaptive"}
        )
        self.client = boto3.client("bedrock-runtime", config=self.config)
        
        # LangChain integration
        self.langchain_llm = ChatBedrock(
            model_id=self.CLAUDE_35_SONNET_BEDROCK,
            client=self.client,
            model_kwargs={"temperature": 0.7, "max_tokens": 1024}
        )
        
        self.langchain_embeddings = BedrockEmbeddings(
            model_id=self.TITAN_EMBED_TEXT_V2,
            client=self.client
        )
    
    def invoke_claude(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = 1024
    ) -> str:
        """Invoke Claude model on Bedrock."""
        model = model or self.CLAUDE_35_SONNET_BEDROCK
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        })
        
        response = self.client.invoke_model(
            modelId=model,
            body=body
        )
        
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
    
    def invoke_titan(self, prompt: str, model: str = None) -> str:
        """Invoke Amazon Titan model."""
        model = model or self.TITAN_TEXT_EXPRESS
        
        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 1024,
                "temperature": 0.7
            }
        })
        
        response = self.client.invoke_model(modelId=model, body=body)
        result = json.loads(response["body"].read())
        return result["results"][0]["outputText"]
    
    def invoke_llama(self, prompt: str, model: str = None) -> str:
        """Invoke Meta Llama model on Bedrock."""
        model = model or self.LLAMA_31_70B
        
        body = json.dumps({
            "prompt": prompt,
            "max_gen_len": 1024,
            "temperature": 0.7
        })
        
        response = self.client.invoke_model(modelId=model, body=body)
        result = json.loads(response["body"].read())
        return result["generation"]
    
    def create_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """Create embeddings using Titan."""
        model = model or self.TITAN_EMBED_TEXT_V2
        embeddings = []
        
        for text in texts:
            body = json.dumps({"inputText": text})
            response = self.client.invoke_model(modelId=model, body=body)
            result = json.loads(response["body"].read())
            embeddings.append(result["embedding"])
        
        return embeddings


# =============================================================================
# LlamaIndex Integration
# =============================================================================

class LlamaIndexManager:
    """Manager for LlamaIndex operations."""
    
    def __init__(
        self,
        llm_provider: str = "openai",
        embedding_provider: str = "openai"
    ):
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self._setup_llm()
        self._setup_embeddings()
    
    def _setup_llm(self):
        """Setup LLM based on provider."""
        if self.llm_provider == "openai":
            self.llm = LlamaOpenAI(model="gpt-4o", temperature=0.7)
        elif self.llm_provider == "anthropic":
            self.llm = LlamaAnthropic(model="claude-3-5-sonnet-20241022")
        Settings.llm = self.llm
    
    def _setup_embeddings(self):
        """Setup embeddings based on provider."""
        if self.embedding_provider == "openai":
            self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        elif self.embedding_provider == "huggingface":
            self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.embed_model = self.embed_model
    
    def create_index_from_documents(
        self,
        documents: List[LlamaDocument]
    ) -> VectorStoreIndex:
        """Create a vector store index from documents."""
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        Settings.node_parser = splitter
        
        index = VectorStoreIndex.from_documents(documents)
        return index
    
    def create_query_engine(self, index: VectorStoreIndex) -> RetrieverQueryEngine:
        """Create a query engine from an index."""
        retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
        query_engine = RetrieverQueryEngine(retriever=retriever)
        return query_engine
    
    def query(self, index: VectorStoreIndex, query_text: str) -> str:
        """Query the index."""
        query_engine = index.as_query_engine()
        response = query_engine.query(query_text)
        return str(response)


# =============================================================================
# spaCy NLP
# =============================================================================

class SpacyNLPProcessor:
    """NLP processor using spaCy."""
    
    # Available models
    EN_CORE_WEB_SM = "en_core_web_sm"
    EN_CORE_WEB_MD = "en_core_web_md"
    EN_CORE_WEB_LG = "en_core_web_lg"
    EN_CORE_WEB_TRF = "en_core_web_trf"  # Transformer-based
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.EN_CORE_WEB_SM
        try:
            self.nlp = spacy.load(self.model_name)
        except OSError:
            print(f"Model {self.model_name} not found. Please download it.")
            self.nlp = None
    
    def process_text(self, text: str) -> Doc:
        """Process text and return spaCy Doc."""
        return self.nlp(text)
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from text."""
        doc = self.nlp(text)
        return [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]
    
    def extract_noun_phrases(self, text: str) -> List[str]:
        """Extract noun phrases from text."""
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]
    
    def get_pos_tags(self, text: str) -> List[Tuple[str, str]]:
        """Get part-of-speech tags."""
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]
    
    def get_dependencies(self, text: str) -> List[Dict[str, str]]:
        """Get dependency parse information."""
        doc = self.nlp(text)
        return [
            {"text": token.text, "dep": token.dep_, "head": token.head.text}
            for token in doc
        ]
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        return doc1.similarity(doc2)
    
    def create_matcher(self, patterns: List[Dict]) -> Matcher:
        """Create a rule-based matcher."""
        matcher = Matcher(self.nlp.vocab)
        for pattern in patterns:
            matcher.add(pattern["name"], [pattern["pattern"]])
        return matcher


# =============================================================================
# Hugging Face Transformers Manager
# =============================================================================

class TransformersManager:
    """Manager for Hugging Face Transformers models."""
    
    # BERT family
    BERT_BASE = "bert-base-uncased"
    BERT_LARGE = "bert-large-uncased"
    ROBERTA_BASE = "roberta-base"
    ROBERTA_LARGE = "roberta-large"
    DISTILBERT = "distilbert-base-uncased"
    ALBERT_BASE = "albert-base-v2"
    ELECTRA_BASE = "google/electra-base-discriminator"
    DEBERTA_BASE = "microsoft/deberta-base"
    
    # GPT family
    GPT2 = "gpt2"
    GPT2_MEDIUM = "gpt2-medium"
    GPT2_LARGE = "gpt2-large"
    GPT2_XL = "gpt2-xl"
    
    # T5 family
    T5_SMALL = "t5-small"
    T5_BASE = "t5-base"
    T5_LARGE = "t5-large"
    FLAN_T5_BASE = "google/flan-t5-base"
    FLAN_T5_LARGE = "google/flan-t5-large"
    
    # Modern LLMs
    LLAMA_2_7B = "meta-llama/Llama-2-7b-hf"
    LLAMA_2_13B = "meta-llama/Llama-2-13b-hf"
    LLAMA_3_8B = "meta-llama/Meta-Llama-3-8B"
    LLAMA_31_8B = "meta-llama/Llama-3.1-8B"
    MISTRAL_7B = "mistralai/Mistral-7B-v0.1"
    MISTRAL_7B_INSTRUCT = "mistralai/Mistral-7B-Instruct-v0.2"
    MIXTRAL_8X7B = "mistralai/Mixtral-8x7B-v0.1"
    PHI_3_MINI = "microsoft/Phi-3-mini-4k-instruct"
    GEMMA_2B = "google/gemma-2b"
    GEMMA_7B = "google/gemma-7b"
    QWEN2_7B = "Qwen/Qwen2-7B"
    FALCON_7B = "tiiuae/falcon-7b"
    FALCON_40B = "tiiuae/falcon-40b"
    
    # Vision models
    VIT_BASE = "google/vit-base-patch16-224"
    CLIP_BASE = "openai/clip-vit-base-patch32"
    
    # Sentence Transformers
    SENTENCE_BERT = "sentence-transformers/all-MiniLM-L6-v2"
    SENTENCE_MPNET = "sentence-transformers/all-mpnet-base-v2"
    BGE_SMALL = "BAAI/bge-small-en-v1.5"
    BGE_BASE = "BAAI/bge-base-en-v1.5"
    BGE_LARGE = "BAAI/bge-large-en-v1.5"
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.accelerator = Accelerator()
    
    def load_text_classifier(self, model_name: str, num_labels: int = 2):
        """Load a model for text classification."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)
        return tokenizer, model
    
    def load_causal_lm(self, model_name: str):
        """Load a causal language model."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        return tokenizer, model
    
    def load_seq2seq(self, model_name: str):
        """Load a sequence-to-sequence model."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        return tokenizer, model
    
    def load_sentence_transformer(self, model_name: str = None) -> SentenceTransformer:
        """Load a sentence transformer model."""
        model_name = model_name or self.SENTENCE_MPNET
        return SentenceTransformer(model_name, device=self.device)
    
    def create_pipeline(self, task: str, model_name: str = None):
        """Create a Hugging Face pipeline."""
        return pipeline(task, model=model_name, device=0 if self.device == "cuda" else -1)
    
    def generate_text(self, model_name: str, prompt: str, max_length: int = 100) -> str:
        """Generate text using a causal LM."""
        tokenizer, model = self.load_causal_lm(model_name)
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def get_embeddings(self, texts: List[str], model_name: str = None) -> np.ndarray:
        """Get embeddings using sentence transformers."""
        model = self.load_sentence_transformer(model_name)
        return model.encode(texts)


# =============================================================================
# PyTorch Models
# =============================================================================

class PyTorchTextClassifier(nn.Module):
    """Simple PyTorch text classifier using embeddings."""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super(PyTorchTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class PyTorchCNN(nn.Module):
    """Convolutional Neural Network for image classification."""
    
    def __init__(self, num_classes: int = 10):
        super(PyTorchCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PyTorchTransformer(nn.Module):
    """Transformer model for sequence processing."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        num_classes: int = 2
    ):
        super(PyTorchTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


class PyTorchModelManager:
    """Manager for PyTorch pretrained models."""
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_resnet(self, num_classes: int = 1000) -> nn.Module:
        """Load pretrained ResNet50."""
        model = torchvision_models.resnet50(weights=torchvision_models.ResNet50_Weights.IMAGENET1K_V2)
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(self.device)
    
    def load_vgg(self) -> nn.Module:
        """Load pretrained VGG16."""
        return torchvision_models.vgg16(weights=torchvision_models.VGG16_Weights.IMAGENET1K_V1).to(self.device)
    
    def load_efficientnet(self) -> nn.Module:
        """Load pretrained EfficientNet."""
        return torchvision_models.efficientnet_b0(weights=torchvision_models.EfficientNet_B0_Weights.IMAGENET1K_V1).to(self.device)
    
    def get_transforms(self) -> transforms.Compose:
        """Get standard image transforms."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# =============================================================================
# TensorFlow / Keras Models
# =============================================================================

class TensorFlowModelBuilder:
    """Builder for TensorFlow/Keras models."""
    
    # TensorFlow Hub URLs
    BERT_EN = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
    UNIVERSAL_SENTENCE_ENCODER = "https://tfhub.dev/google/universal-sentence-encoder/4"
    MOBILENET_V2_HUB = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
    EFFICIENTNET_B0_HUB = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
    
    def __init__(self):
        gpus = tf.config.list_physical_devices('GPU')
        print(f"TensorFlow GPUs: {len(gpus)}")
    
    def build_text_classifier(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_length: int,
        num_classes: int
    ) -> keras.Model:
        """Build a text classification model."""
        model = keras.Sequential([
            layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def build_cnn(self, num_classes: int = 10) -> keras.Model:
        """Build a CNN for image classification."""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def build_lstm(self, vocab_size: int, embedding_dim: int, lstm_units: int, num_classes: int) -> keras.Model:
        """Build an LSTM model."""
        model = keras.Sequential([
            layers.Embedding(vocab_size, embedding_dim),
            layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(lstm_units)),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def load_pretrained_resnet(self, num_classes: int = 1000) -> keras.Model:
        """Load pretrained ResNet50."""
        base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = layers.GlobalAveragePooling2D()(base.output)
        x = layers.Dense(num_classes, activation='softmax')(x)
        return keras.Model(inputs=base.input, outputs=x)
    
    def load_pretrained_mobilenet(self, num_classes: int = 1000) -> keras.Model:
        """Load pretrained MobileNetV2."""
        base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = layers.GlobalAveragePooling2D()(base.output)
        x = layers.Dense(num_classes, activation='softmax')(x)
        return keras.Model(inputs=base.input, outputs=x)
    
    def load_hub_model(self, hub_url: str) -> keras.layers.Layer:
        """Load a model from TensorFlow Hub."""
        return hub.KerasLayer(hub_url, trainable=False)


# =============================================================================
# Scikit-learn Pipeline
# =============================================================================

class SklearnPipeline:
    """Traditional ML pipeline using scikit-learn."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            "random_forest": RandomForestClassifier(n_estimators=100),
            "gradient_boosting": GradientBoostingClassifier(),
            "logistic_regression": LogisticRegression(),
            "svm": SVC(probability=True),
            "knn": KNeighborsClassifier(),
            "naive_bayes": GaussianNB(),
            "decision_tree": DecisionTreeClassifier(),
            "adaboost": AdaBoostClassifier()
        }
    
    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = "random_forest",
        test_size: float = 0.2
    ) -> Dict:
        """Train and evaluate a model."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model = self.models[model_name]
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
    
    def create_text_pipeline(self) -> Pipeline:
        """Create a text classification pipeline."""
        return Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', MultinomialNB())
        ])
    
    def cluster_data(self, X: np.ndarray, n_clusters: int = 5, method: str = "kmeans") -> np.ndarray:
        """Cluster data using various methods."""
        X_scaled = self.scaler.fit_transform(X)
        
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters)
        elif method == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        elif method == "hierarchical":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        
        return clusterer.fit_predict(X_scaled)
    
    def reduce_dimensions(self, X: np.ndarray, n_components: int = 2, method: str = "pca") -> np.ndarray:
        """Reduce dimensionality."""
        if method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "svd":
            reducer = TruncatedSVD(n_components=n_components)
        
        return reducer.fit_transform(X)


# =============================================================================
# Data Analysis with Pandas/NumPy/Seaborn
# =============================================================================

class DataAnalyzer:
    """Data analysis utilities using pandas, numpy, and seaborn."""
    
    def __init__(self):
        sns.set_theme(style="whitegrid")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from various formats."""
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            return pd.read_json(filepath)
        elif filepath.endswith('.parquet'):
            return pd.read_parquet(filepath)
        elif filepath.endswith('.xlsx'):
            return pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict:
        """Comprehensive dataframe analysis."""
        return {
            "shape": df.shape,
            "dtypes": df.dtypes.to_dict(),
            "missing": df.isnull().sum().to_dict(),
            "describe": df.describe().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }
    
    def compute_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation matrix."""
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df.corr()
    
    def plot_distribution(self, df: pd.DataFrame, column: str, figsize: Tuple = (10, 6)):
        """Plot distribution of a column."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        sns.histplot(df[column], kde=True, ax=axes[0])
        axes[0].set_title(f'Distribution of {column}')
        
        # Box plot
        sns.boxplot(y=df[column], ax=axes[1])
        axes[1].set_title(f'Box Plot of {column}')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, figsize: Tuple = (12, 10)):
        """Plot correlation heatmap."""
        corr = self.compute_correlations(df)
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Heatmap')
        return fig
    
    def plot_pairplot(self, df: pd.DataFrame, hue: str = None):
        """Create pairplot."""
        return sns.pairplot(df, hue=hue)
    
    def numpy_operations(self, data: np.ndarray) -> Dict:
        """Various numpy operations."""
        return {
            "mean": np.mean(data),
            "std": np.std(data),
            "median": np.median(data),
            "min": np.min(data),
            "max": np.max(data),
            "percentiles": np.percentile(data, [25, 50, 75]),
            "shape": data.shape,
            "dtype": str(data.dtype)
        }
    
    def create_interactive_plot(self, df: pd.DataFrame, x: str, y: str, color: str = None):
        """Create interactive plot with Plotly."""
        fig = px.scatter(df, x=x, y=y, color=color, title=f"{y} vs {x}")
        return fig


# =============================================================================
# Vector Database Manager
# =============================================================================

class VectorDBManager:
    """Manager for vector databases."""
    
    def __init__(self):
        self.chroma_client = chromadb.Client(ChromaSettings(anonymized_telemetry=False))
    
    def create_chroma_collection(self, name: str):
        """Create a ChromaDB collection."""
        return self.chroma_client.create_collection(name=name)
    
    def add_to_collection(
        self,
        collection,
        documents: List[str],
        embeddings: List[List[float]],
        ids: List[str],
        metadatas: List[Dict] = None
    ):
        """Add documents to collection."""
        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
    
    def query_collection(
        self,
        collection,
        query_embeddings: List[List[float]],
        n_results: int = 5
    ):
        """Query a collection."""
        return collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main entry point demonstrating all AI/ML capabilities."""
    print("=" * 80)
    print("   Comprehensive AI/ML Application - AIBOM Test Sample")
    print("=" * 80)
    
    # =========================================================================
    # SECTION 1: OpenAI
    # =========================================================================
    print("\n" + "-" * 80)
    print("SECTION 1: OpenAI Models")
    print("-" * 80)
    
    print("\n[1.1] Available OpenAI Models:")
    print(f"      Chat: {OpenAIClient.GPT_4O}, {OpenAIClient.GPT_4_TURBO}, {OpenAIClient.GPT_35_TURBO}")
    print(f"      Embeddings: {OpenAIClient.TEXT_EMBEDDING_3_SMALL}, {OpenAIClient.TEXT_EMBEDDING_3_LARGE}")
    print(f"      Image: {OpenAIClient.DALL_E_3}, {OpenAIClient.DALL_E_2}")
    print(f"      Audio: {OpenAIClient.WHISPER_1}, {OpenAIClient.TTS_1}")
    
    # =========================================================================
    # SECTION 2: Anthropic
    # =========================================================================
    print("\n" + "-" * 80)
    print("SECTION 2: Anthropic Claude Models")
    print("-" * 80)
    
    print("\n[2.1] Available Claude Models:")
    print(f"      Claude 3.5: {AnthropicClient.CLAUDE_35_SONNET}, {AnthropicClient.CLAUDE_35_HAIKU}")
    print(f"      Claude 3: {AnthropicClient.CLAUDE_3_OPUS}, {AnthropicClient.CLAUDE_3_SONNET}, {AnthropicClient.CLAUDE_3_HAIKU}")
    print(f"      Legacy: {AnthropicClient.CLAUDE_2}, {AnthropicClient.CLAUDE_INSTANT}")
    
    # =========================================================================
    # SECTION 3: AWS Bedrock
    # =========================================================================
    print("\n" + "-" * 80)
    print("SECTION 3: AWS Bedrock Models")
    print("-" * 80)
    
    print("\n[3.1] Available Bedrock Models:")
    print(f"      Claude: {BedrockClient.CLAUDE_35_SONNET_BEDROCK}")
    print(f"      Titan: {BedrockClient.TITAN_TEXT_EXPRESS}, {BedrockClient.TITAN_EMBED_TEXT_V2}")
    print(f"      Llama: {BedrockClient.LLAMA_31_8B}, {BedrockClient.LLAMA_31_70B}")
    print(f"      Mistral: {BedrockClient.MISTRAL_7B}, {BedrockClient.MIXTRAL_8X7B}")
    print(f"      Cohere: {BedrockClient.COHERE_COMMAND}, {BedrockClient.COHERE_EMBED_ENGLISH}")
    
    # =========================================================================
    # SECTION 4: Hugging Face Transformers
    # =========================================================================
    print("\n" + "-" * 80)
    print("SECTION 4: Hugging Face Transformers")
    print("-" * 80)
    
    print("\n[4.1] BERT Family:")
    print(f"      {TransformersManager.BERT_BASE}, {TransformersManager.ROBERTA_BASE}, {TransformersManager.DISTILBERT}")
    
    print("\n[4.2] GPT Family:")
    print(f"      {TransformersManager.GPT2}, {TransformersManager.GPT2_MEDIUM}, {TransformersManager.GPT2_XL}")
    
    print("\n[4.3] Modern LLMs:")
    print(f"      LLaMA: {TransformersManager.LLAMA_31_8B}")
    print(f"      Mistral: {TransformersManager.MISTRAL_7B_INSTRUCT}, {TransformersManager.MIXTRAL_8X7B}")
    print(f"      Phi: {TransformersManager.PHI_3_MINI}")
    print(f"      Gemma: {TransformersManager.GEMMA_2B}, {TransformersManager.GEMMA_7B}")
    print(f"      Qwen: {TransformersManager.QWEN2_7B}")
    print(f"      Falcon: {TransformersManager.FALCON_7B}")
    
    print("\n[4.4] Sentence Transformers:")
    print(f"      {TransformersManager.SENTENCE_MPNET}, {TransformersManager.BGE_LARGE}")
    
    # =========================================================================
    # SECTION 5: LlamaIndex
    # =========================================================================
    print("\n" + "-" * 80)
    print("SECTION 5: LlamaIndex")
    print("-" * 80)
    
    print("\n[5.1] LlamaIndex components configured")
    print("      LLM providers: OpenAI, Anthropic")
    print("      Embedding providers: OpenAI, HuggingFace")
    
    # =========================================================================
    # SECTION 6: spaCy NLP
    # =========================================================================
    print("\n" + "-" * 80)
    print("SECTION 6: spaCy NLP")
    print("-" * 80)
    
    print("\n[6.1] Available spaCy Models:")
    print(f"      {SpacyNLPProcessor.EN_CORE_WEB_SM}")
    print(f"      {SpacyNLPProcessor.EN_CORE_WEB_MD}")
    print(f"      {SpacyNLPProcessor.EN_CORE_WEB_LG}")
    print(f"      {SpacyNLPProcessor.EN_CORE_WEB_TRF} (Transformer-based)")
    
    # =========================================================================
    # SECTION 7: PyTorch
    # =========================================================================
    print("\n" + "-" * 80)
    print("SECTION 7: PyTorch")
    print("-" * 80)
    
    print("\n[7.1] PyTorch Custom Models:")
    print("      PyTorchTextClassifier, PyTorchCNN, PyTorchTransformer")
    
    print("\n[7.2] PyTorch Pretrained Models:")
    print("      ResNet50, VGG16, EfficientNet")
    
    # =========================================================================
    # SECTION 8: TensorFlow
    # =========================================================================
    print("\n" + "-" * 80)
    print("SECTION 8: TensorFlow / Keras")
    print("-" * 80)
    
    print("\n[8.1] Keras Models:")
    print("      Text Classifier, CNN, LSTM")
    
    print("\n[8.2] TensorFlow Hub Models:")
    print(f"      BERT: {TensorFlowModelBuilder.BERT_EN}")
    print(f"      USE: {TensorFlowModelBuilder.UNIVERSAL_SENTENCE_ENCODER}")
    
    # =========================================================================
    # SECTION 9: Scikit-learn
    # =========================================================================
    print("\n" + "-" * 80)
    print("SECTION 9: Scikit-learn")
    print("-" * 80)
    
    print("\n[9.1] Classification Models:")
    print("      RandomForest, GradientBoosting, LogisticRegression, SVM, KNN")
    
    print("\n[9.2] Clustering:")
    print("      KMeans, DBSCAN, AgglomerativeClustering")
    
    print("\n[9.3] Dimensionality Reduction:")
    print("      PCA, TruncatedSVD")
    
    # =========================================================================
    # SECTION 10: Data Science Stack
    # =========================================================================
    print("\n" + "-" * 80)
    print("SECTION 10: Data Science (NumPy, Pandas, Seaborn)")
    print("-" * 80)
    
    print("\n[10.1] NumPy operations available")
    print("\n[10.2] Pandas DataFrame analysis")
    print("\n[10.3] Seaborn/Matplotlib/Plotly visualizations")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("   AIBOM Test Sample Complete!")
    print("=" * 80)
    print("\nThis application demonstrates:")
    print("  ┌──────────────────────────────────────────────────────────────────────────┐")
    print("  │ LLM PROVIDERS                                                            │")
    print("  │   • OpenAI: GPT-4, GPT-4o, GPT-3.5, DALL-E, Whisper                     │")
    print("  │   • Anthropic: Claude 3.5 Sonnet/Haiku, Claude 3 Opus/Sonnet/Haiku      │")
    print("  │   • AWS Bedrock: Claude, Titan, Llama, Mistral, Cohere                  │")
    print("  ├──────────────────────────────────────────────────────────────────────────┤")
    print("  │ FRAMEWORKS                                                               │")
    print("  │   • LangChain: Chains, Memory, RAG                                       │")
    print("  │   • LlamaIndex: Vector stores, Query engines                             │")
    print("  ├──────────────────────────────────────────────────────────────────────────┤")
    print("  │ DEEP LEARNING                                                            │")
    print("  │   • PyTorch: Custom models, Pretrained (ResNet, VGG, EfficientNet)      │")
    print("  │   • TensorFlow/Keras: Sequential, CNN, LSTM, TF Hub                     │")
    print("  │   • Transformers: BERT, GPT-2, LLaMA, Mistral, Phi, Gemma, Qwen, Falcon │")
    print("  ├──────────────────────────────────────────────────────────────────────────┤")
    print("  │ NLP                                                                      │")
    print("  │   • spaCy: NER, POS tagging, Dependency parsing                         │")
    print("  │   • Sentence Transformers: Embeddings                                    │")
    print("  ├──────────────────────────────────────────────────────────────────────────┤")
    print("  │ TRADITIONAL ML                                                           │")
    print("  │   • Scikit-learn: Classification, Clustering, Dimensionality Reduction  │")
    print("  ├──────────────────────────────────────────────────────────────────────────┤")
    print("  │ DATA SCIENCE                                                             │")
    print("  │   • NumPy, Pandas, Seaborn, Matplotlib, Plotly                          │")
    print("  └──────────────────────────────────────────────────────────────────────────┘")
    print("=" * 80)


if __name__ == "__main__":
    main()
