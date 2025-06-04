from flask import Flask, request, jsonify
from neo4j import GraphDatabase
from flask_cors import CORS
from collections import Counter
import jieba
from py2neo import Graph
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, Linear
from torch_geometric.data import HeteroData
import pickle

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

    
    relation_map = {
        "并发症": "acompany_with",
        "症状": "has_symptom",
        "饮食": "recommand_eat|no_eat|do_eat",
        "药物": "recommand_drug|common_drug",
        "检查": "need_check",
        "科室": "belongs_to"
    }

    relation = relation_map.get(graph_type)
    if not relation:
        return jsonify({"nodes": [], "edges": []})

    # 构建关系查询（多个关系时用 | 拼接）
    query = f"""
    MATCH (d:Disease {{name: $name}})-[r:{relation}]-(n)
    RETURN DISTINCT d, r, n
    """

    with driver.session() as session:
        result = session.run(query, name=name)
        nodes = {}
        edges = []
        seen_edges = set() 

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
            
            edge_key = (d.id, n.id)  # 忽略关系类型
            if edge_key not in seen_edges:
                edges.append({
                    "data": {
                        "source": str(d.id),
                        "target": str(n.id),
                        "label": r.type
                    }
                })
                seen_edges.add(edge_key)


            # edges.append({
            #     "data": {
            #         "source": str(d.id),
            #         "target": str(n.id),
            #         "label": r.type
            #     }
            # })

        return jsonify({
            "nodes": list(nodes.values()),
            "edges": edges
        })


# ========== 模型结构 ==========
class SymptomMLP(nn.Module):
    def __init__(self, in_dim=768, hidden=128, out_dim=54):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.fc(x)

# ========== 加载模型与数据 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mlp_model = SymptomMLP().to(device)
mlp_model.load_state_dict(torch.load("mlp_model.pt", map_location=device))
mlp_model.eval()

with open("symptom_name2idx.pkl", "rb") as f:
    symptom_name2idx = pickle.load(f)
with open("department_idx2name.pkl", "rb") as f:
    dept_idx2name = pickle.load(f)

symptom_feat = torch.load("symptom_feat.pt")

# ========== 推理函数 ==========
@torch.no_grad()
def get_symptom_vector(name):
    if name in symptom_name2idx:
        idx = symptom_name2idx[name]
        return symptom_feat[idx].unsqueeze(0)
    else:
        print(f"❗症状 `{name}` 未命中缓存")
        return None

@torch.no_grad()
def predict_with_mlp(symptom_names, k=5):
    vecs = [get_symptom_vector(name) for name in symptom_names]
    vecs = [v for v in vecs if v is not None]

    if not vecs:
        return []

    input_tensor = torch.mean(torch.cat(vecs, dim=0), dim=0, keepdim=True).to(device)
    logits = mlp_model(input_tensor)
    probs = F.softmax(logits, dim=-1)
    topk = torch.topk(probs, k=k)

    return [(dept_idx2name[int(idx)], float(score)) for idx, score in zip(topk.indices[0], topk.values[0])]

# ========== Flask 接口 ==========
@app.route('/api/predict', methods=['POST'])
def predict_api():
    input_data = request.get_json()
    input_symptoms = input_data.get('symptoms', [])

    if not input_symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    results = predict_with_mlp(input_symptoms)

    if not results:
        return jsonify({"error": "No valid symptoms found"}), 400

    response = [
        {"department": dept, "probability": round(prob, 4)}
        for dept, prob in results
    ]
    return jsonify(response)

# ========== 启动服务 ==========
if __name__ == "__main__":
    app.run(debug=True)