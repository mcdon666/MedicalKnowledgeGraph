from flask import Flask, request, jsonify
from neo4j import GraphDatabase
from flask_cors import CORS
from collections import Counter
import jieba
from py2neo import Graph

from question_classifier import *
from question_parser import *
from answer_search import *


app = Flask(__name__)
CORS(app)

# 替换为你自己的 Neo4j 地址和密码
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))
graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))

# ========== 疾病属性词云 ==========
@app.route("/api/wordcloud", methods=["GET"])
def get_wordcloud():
    
    name = request.args.get("name")
    if not name:
        return jsonify([])

    query = "MATCH (d:Disease {name: $name}) RETURN properties(d) AS props"
    result = graph.run(query, name=name).data()

    if not result:
        return jsonify([])

    props = result[0]['props']
    text = ""

    # 拼接所有属性文本
    for value in props.values():
        if isinstance(value, str):
            text += value
        elif isinstance(value, list):
            text += "".join(str(v) for v in value)
        else:
            text += str(value)

    # 中文分词
    words = jieba.lcut(text)
    stopwords = set(["包括", "具有", "疾病", "可能", "患者"])
    filtered = [w for w in words if len(w) > 1 and w not in stopwords]
    freq = Counter(filtered)

    # 格式化为 ECharts 所需格式
    data = [{"name": k, "value": v} for k, v in freq.items()]
    return jsonify(data)

# ========== 问答系统 ==========
class ChatBotGraph:
    def __init__(self):
        self.classifier = QuestionClassifier()
        self.parser = QuestionPaser()
        self.searcher = AnswerSearcher()

    def chat_main(self, sent):
        default_answer = '您好，本问答内容仅供参考，如有不适请及时就医。'
        res_classify = self.classifier.classify(sent)
        if not res_classify:
            return default_answer
        res_sql = self.parser.parser_main(res_classify)
        final_answers = self.searcher.search_main(res_sql)
        return '\n'.join(final_answers) if final_answers else default_answer

chatbot = ChatBotGraph()

@app.route('/api/chat', methods=['POST'])
def chat_api():
    user_input = request.json.get('question')
    answer = chatbot.chat_main(user_input)
    return jsonify({"answer": answer})


# ========== 图谱查询 ==========
@app.route("/api/graph")
def get_subgraph():
    name = request.args.get("name")
    graph_type = request.args.get("type")

    if not name or not graph_type:
        return jsonify({"nodes": [], "edges": []})

    # 定义关系映射
    relation_map = {
        "并发症": "acompany_with",
        "症状": "has_symptom",
        "饮食": "recommend_eat|not_eat|do_eat",
        "药物": "recommend_drug|common_drug"
    }

    relation = relation_map.get(graph_type)
    if not relation:
        return jsonify({"nodes": [], "edges": []})

    # 构建关系查询（多个关系时用 | 拼接）
    query = f"""
    MATCH (d:Disease {{name: $name}})-[r:{relation}]-(n)
    RETURN d, r, n
    """

    with driver.session() as session:
        result = session.run(query, name=name)
        nodes = {}
        edges = []

        for record in result:
            d = record["d"]
            n = record["n"]
            r = record["r"]

            for node in [d, n]:
                if node.id not in nodes:
                    nodes[node.id] = {
                        "data": {
                            "id": str(node.id),
                            "label": node.get("name", "Unnamed"),
                            "type": list(node.labels)[0]
                        }
                    }

            edges.append({
                "data": {
                    "source": str(d.id),
                    "target": str(n.id),
                    "label": r.type
                }
            })

        return jsonify({
            "nodes": list(nodes.values()),
            "edges": edges
        })


# ========== 症状预测疾病 ==========

# ========== 启动服务 ==========
if __name__ == "__main__":
    app.run(debug=True)