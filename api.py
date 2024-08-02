import os
import asyncio
import time
import uuid
import json
import re
import pandas as pd
import tiktoken
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager


# GraphRAG 相关导入
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.query.structured_search.base import SearchResult

# 加载 .env 到环境变量
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置常量和配置
INPUT_DIR = os.getenv('INPUT_DIR')
LANCEDB_URI = f"{INPUT_DIR}/lancedb"
COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"
COMMUNITY_LEVEL = 2
PORT = 8012

# 全局变量，用于存储搜索引擎和问题生成器
local_search_engine: LocalSearch
global_search_engine: GlobalSearch
question_generator: LocalQuestionGen

# 数据模型
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Usage
    system_fingerprint: Optional[str] = None


async def setup_llm_and_embedder():
    """
    设置语言模型（LLM）和嵌入模型
    """
    logger.info("正在设置LLM和嵌入器")

    # 获取API密钥和基础URL
    api_key = os.environ.get("GRAPHRAG_API_KEY", "YOUR_API_KEY")
    api_key_embedding = os.environ.get("GRAPHRAG_EMBEDDING_API_KEY", api_key)
    api_base = os.environ.get("GRAPHRAG_API_BASE", "https://api.openai.com/v1")
    api_base_embedding = os.environ.get("GRAPHRAG_EMBEDDING_API_BASE", "https://api.openai.com/v1")

    # 获取模型名称
    llm_model = os.environ.get("GRAPHRAG_LLM_MODEL", "EGPT-3.0")
    embedding_model = os.environ.get("GRAPHRAG_EMBEDDING_MODEL", "text-embedding-3-small")

    # 检查API密钥是否存在
    if api_key == "YOUR_API_KEY":
        logger.error("环境变量中未找到有效的GRAPHRAG_API_KEY")
        raise ValueError("GRAPHRAG_API_KEY未正确设置")

    # 初始化ChatOpenAI实例
    llm = ChatOpenAI(
        api_key=api_key,
        api_base=api_base,
        model=llm_model,
        api_type=OpenaiApiType.OpenAI,
        max_retries=2,
    )

    # 初始化token编码器
    # token_encoder = tiktoken.get_encoding("cl100k_base")
    token_encoder = tiktoken.get_encoding("o200k_base")

    # 初始化文本嵌入模型
    text_embedder = OpenAIEmbedding(
        api_key=api_key_embedding,
        api_base=api_base_embedding,
        api_type=OpenaiApiType.OpenAI,
        model=embedding_model,
        deployment_name=embedding_model,
        max_retries=2,
    )


    logger.info("LLM和嵌入器设置完成")
    return llm, token_encoder, text_embedder


async def load_context():
    """
    加载上下文数据，包括实体、关系、报告、文本单元和协变量
    """
    logger.info("正在加载上下文数据")
    try:
        entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
        entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
        entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

        description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")
        description_embedding_store.connect(db_uri=LANCEDB_URI)
        store_entity_semantic_embeddings(entities=entities, vectorstore=description_embedding_store)

        relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
        relationships = read_indexer_relationships(relationship_df)

        report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
        reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)

        text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
        text_units = read_indexer_text_units(text_unit_df)

        covariate_df = pd.read_parquet(f"{INPUT_DIR}/{COVARIATE_TABLE}.parquet")
        claims = read_indexer_covariates(covariate_df)
        logger.info(f"声明记录数: {len(claims)}")
        covariates = {"claims": claims}

        logger.info("上下文数据加载完成")
        return entities, relationships, reports, text_units, description_embedding_store, covariates
    except Exception as e:
        logger.error(f"加载上下文数据时出错: {str(e)}")
        raise


async def setup_search_engines(llm, token_encoder, text_embedder, entities, relationships, reports, text_units,
                               description_embedding_store, covariates):
    """
    设置本地搜索引擎和全局搜索引擎
    """
    logger.info("正在设置搜索引擎")

    # 设置本地搜索引擎
    local_context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates=covariates,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )

    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        "max_tokens": 12_000,
    }

    local_llm_params = {
        "max_tokens": 2_000,
        "temperature": 0.0,
    }

    local_search_engine = LocalSearch(
        llm=llm,
        context_builder=local_context_builder,
        token_encoder=token_encoder,
        llm_params=local_llm_params,
        context_builder_params=local_context_params,
        response_type="multiple paragraphs",
    )

    # 设置全局搜索引擎
    global_context_builder = GlobalCommunityContext(
        community_reports=reports,
        entities=entities,
        token_encoder=token_encoder,
    )

    global_context_builder_params = {
        "use_community_summary": False,
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,
        "context_name": "Reports",
    }

    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    reduce_llm_params = {
        "max_tokens": 2000,
        "temperature": 0.0,
    }

    global_search_engine = GlobalSearch(
        llm=llm,
        context_builder=global_context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,
        json_mode=True,
        context_builder_params=global_context_builder_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",
    )

    logger.info("搜索引擎设置完成")
    return local_search_engine, global_search_engine, local_context_builder, local_llm_params, local_context_params


def format_response(response):
    """
    格式化响应，添加适当的换行和段落分隔。
    """
    paragraphs = re.split(r'\n{2,}', response)

    formatted_paragraphs = []
    for para in paragraphs:
        if '```' in para:
            parts = para.split('```')
            for i, part in enumerate(parts):
                if i % 2 == 1:  # 这是代码块
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            para = ''.join(parts)
        else:
            para = para.replace('. ', '.\n')

        formatted_paragraphs.append(para.strip())

    return '\n\n'.join(formatted_paragraphs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    global local_search_engine, global_search_engine, question_generator
    try:
        logger.info("正在初始化搜索引擎和问题生成器...")
        llm, token_encoder, text_embedder = await setup_llm_and_embedder()
        entities, relationships, reports, text_units, description_embedding_store, covariates = await load_context()
        local_search_engine, global_search_engine, local_context_builder, local_llm_params, local_context_params = await setup_search_engines(
            llm, token_encoder, text_embedder, entities, relationships, reports, text_units,
            description_embedding_store, covariates
        )

        question_generator = LocalQuestionGen(
            llm=llm,
            context_builder=local_context_builder,
            token_encoder=token_encoder,
            llm_params=local_llm_params,
            context_builder_params=local_context_params,
        )
        logger.info("初始化完成。")
    except Exception as e:
        logger.error(f"初始化过程中出错: {str(e)}")
        raise

    yield

    # 关闭时执行
    logger.info("正在关闭...")


app = FastAPI(lifespan=lifespan)


# 在 chat_completions 函数中添加以下代码

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not local_search_engine or not global_search_engine:
        logger.error("搜索引擎未初始化")
        raise HTTPException(status_code=500, detail="搜索引擎未初始化")

    try:
        logger.info(f"收到聊天完成请求: {request}")
        prompt = request.messages[-1].content
        logger.info(f"处理提示: {prompt}")

        # 根据模型选择使用不同的搜索方法
        if "global" in request.model:
            result: SearchResult = await global_search_engine.asearch(prompt, streaming=request.stream)
        else:  # 默认使用本地搜索
            result: SearchResult = await local_search_engine.asearch(prompt, streaming=request.stream)

        stream = result.response
        # formatted_response = format_response(result.response)
        formatted_response = stream
        logger.info(f"格式化的搜索结果: {formatted_response}")

        # 流式响应和非流式响应的处理保持不变
        if request.stream:
            async def _read_stream_response(stream_response):
                async for part in stream:
                    if token := part.choices[0].delta.content or "":
                        yield token
                # for chunk in stream_response.iter_lines():
                #     chunk_data = bytes.decode(chunk)
                #     print(chunk_data)
                #     logger.debug(chunk_data)
                #     yield chunk_data + '\n'

            # return StreamingResponse(_read_stream_response(stream), media_type='text/event-stream')

            async def generate_stream():
                chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
                # lines = formatted_response.split('\n')
                # for i, line in enumerate(lines):
                async for part in stream:
                    if token := part.choices[0].delta.content or "":
                        chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    #"delta": {"content": token + '\n'}, # if i > 0 else {"role": "assistant", "content": ""},
                                    "delta": {"content": token},
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        await asyncio.sleep(0.05)

                # data: {"id":"chatcmpl-e7048951e7c94e8598b135a9afd92ca2","object":"chat.completion.chunk","created":1722398493,"model":"moonshot-v1-32k","choices":[{"index":0,"delta":{},"finish_reason":"stop","usage":{"prompt_tokens":248,"completion_tokens":86,"total_tokens":334}}]}
                final_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"prompt_tokens": result.prompt_tokens},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            response = ChatCompletionResponse(
                model=request.model,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=Message(role="assistant", content=formatted_response),
                        finish_reason="stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=-1,
                    total_tokens=-1
                )
            )
            logger.info(f"发送响应: {response}")
            return JSONResponse(content=response.dict())

    except Exception as e:
        raise e
        logger.error(f"处理聊天完成时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """
    返回可用模型列表
    """
    logger.info("收到模型列表请求")
    current_time = int(time.time())
    models = [
        {"id": "EGPT-3.0-graph-local", "object": "model", "created": current_time - 100000, "owned_by": "graphrag"},
        {"id": "EGPT-3.0-graph-global", "object": "model", "created": current_time - 95000, "owned_by": "graphrag"},
        # {"id": "graphrag-question-generator:latest", "object": "model", "created": current_time - 90000, "owned_by": "graphrag"},
        # {"id": "gpt-3.5-turbo:latest", "object": "model", "created": current_time - 80000, "owned_by": "openai"},
        # {"id": "text-embedding-3-small:latest", "object": "model", "created": current_time - 70000, "owned_by": "openai"},
        #{"id": "tavily-search:latest", "object": "model", "created": current_time - 85000, "owned_by": "tavily"},
        #{"id": "full-model:latest", "object": "model", "created": current_time - 80000, "owned_by": "combined"}
        # {"id": "combined-local-global:policy", "object": "model", "created": current_time - 80000, "owned_by": "combined"}

    ]

    response = {
        "object": "list",
        "data": models
    }

    logger.info(f"发送模型列表: {response}")
    return JSONResponse(content=response)

if __name__ == "__main__":
    import uvicorn

    logger.info(f"在端口 {PORT} 上启动服务器")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

