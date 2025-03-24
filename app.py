import logging
import os
import time
from typing import Optional

from BCEmbedding import EmbeddingModel, RerankerModel
from flask import Flask, request, jsonify

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 初始化模型
embedding_model: EmbeddingModel = Optional[EmbeddingModel]
reranker_model: RerankerModel = Optional[RerankerModel]
models_loaded = False


def load_models():
    global embedding_model, reranker_model, models_loaded

    if models_loaded:
        return

    # 获取环境变量中的模型路径，如果没有则使用默认值
    embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH", "maidalun1020/bce-embedding-base_v1")
    reranker_model_path = os.getenv("RERANKER_MODEL_PATH", "maidalun1020/bce-reranker-base_v1")

    logger.info(f"开始下载和加载embedding模型: {embedding_model_path}")
    start_time = time.time()
    embedding_model = EmbeddingModel(model_name_or_path=embedding_model_path)
    logger.info(f"Embedding模型加载完成! 耗时: {time.time() - start_time:.2f}秒")

    logger.info(f"开始下载和加载reranker模型: {reranker_model_path}")
    start_time = time.time()
    reranker_model = RerankerModel(model_name_or_path=reranker_model_path)
    logger.info(f"Reranker模型加载完成! 耗时: {time.time() - start_time:.2f}秒")

    models_loaded = True
    logger.info("所有模型加载完成! 服务已准备就绪!")


# 在启动时加载模型，而不是等待第一个请求
load_models()


@app.route('/health', methods=['GET'])
def health():
    global models_loaded
    if models_loaded:
        return jsonify({"status": "healthy", "models_loaded": True})
    else:
        return jsonify({"status": "initializing", "models_loaded": False}), 503


@app.route('/embed', methods=['POST'])
def embed():
    global models_loaded, embedding_model

    if not models_loaded:
        return jsonify({"error": "Models are still loading, please try again later"}), 503

    data = request.json

    if not data or 'input' not in data:
        return jsonify({"error": "Missing 'input' field in request"}), 400

    sentences = data['input']

    try:
        start_time = time.time()
        embeddings = embedding_model.encode(sentences)
        # 将numpy数组转换为列表以便JSON序列化
        embeddings_list = embeddings.tolist()
        logger.info(f"Embedding处理完成，耗时: {time.time() - start_time:.2f}秒")
        return jsonify({"embeddings": embeddings_list})
    except Exception as e:
        logger.error(f"Embedding处理出错: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/rerank', methods=['POST'])
def rerank():
    global models_loaded, reranker_model

    if not models_loaded:
        return jsonify({"error": "Models are still loading, please try again later"}), 503

    data = request.json

    if not data or 'query' not in data or 'documents' not in data:
        return jsonify({"error": "Missing 'query' or 'documents' field in request"}), 400

    query = data['query']
    passages = data['documents']

    if not isinstance(passages, list):
        return jsonify({"error": "'passages' must be a list of strings"}), 400

    try:
        logger.info(f"处理rerank请求，查询: '{query}'，段落数量: {len(passages)}")
        start_time = time.time()
        rerank_results = reranker_model.rerank(query, passages)
        logger.info(f"Rerank处理完成，耗时: {time.time() - start_time:.2f}秒")
        return jsonify({"data": rerank_results})
    except Exception as e:
        logger.error(f"Rerank处理出错: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # 在生产环境中，应该使用适当的WSGI服务器
    port = int(os.getenv("PORT", 5000))
    logger.info(f"启动服务，监听端口: {port}")
    app.run(host='0.0.0.0', port=port)
