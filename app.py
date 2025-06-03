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


# ========== 症状预测疾病 ==========
class Symptom2DiseaseGNN(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = HeteroConv({
            ('symptom', 'has_symptom', 'disease'): GATConv((-1, -1), hidden_dim, add_self_loops=False),
            ('disease', 'rev_has_symptom', 'symptom'): GATConv((-1, -1), hidden_dim, add_self_loops=False),
        }, aggr='sum')
        self.lin = Linear(hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        out = self.lin(x_dict['symptom'])  # 每个 symptom 输出疾病概率
        return out

query = """
MATCH (d:Disease)-[:has_symptom]->(s:Symptom)
RETURN id(d) as did, d.name as dname, id(s) as sid, s.name as sname
"""
results = graph.run(query).data()

symptoms, diseases = {}, {}
edges = []

for row in results:
    sid, sname = row["sid"], row["sname"]
    did, dname = row["did"], row["dname"]
    if sid not in symptoms:
        symptoms[sid] = sname
    if did not in diseases:
        diseases[did] = dname
    edges.append((sid, did))

print(f"✅ 提取完成：{len(symptoms)} 个症状，{len(diseases)} 个疾病，{len(edges)} 条 has_symptom 关系")

# ========== 构建 HeteroData ==========
print("🔧 构建 HeteroData 图对象...")
symptom_id_map = {nid: i for i, nid in enumerate(symptoms)}
disease_id_map = {nid: i for i, nid in enumerate(diseases)}

symptom_to_disease_edge_index = torch.tensor([
    [symptom_id_map[sid] for sid, did in edges],
    [disease_id_map[did] for sid, did in edges]
], dtype=torch.long)

# 手动构造疾病 → 症状的反向边
disease_to_symptom_edge_index = torch.stack([
    symptom_to_disease_edge_index[1],  # target 变 source
    symptom_to_disease_edge_index[0],  # source 变 target
])

# 构建 HeteroData 图
data = HeteroData()
data['symptom'].num_nodes = len(symptoms)
data['disease'].num_nodes = len(diseases)

# 加载特征
symptom_feat = torch.load("symptom_feat.pt")  # shape [num_symptoms, 768]
disease_feat = torch.load("disease_feat.pt")  # shape [num_diseases, 768]

# 绑定到 data 图对象中
data['symptom'].x = symptom_feat
data['disease'].x = disease_feat
print("成功绑定症状与疾病特征向量！")


# 添加双向边
data['symptom', 'has_symptom', 'disease'].edge_index = symptom_to_disease_edge_index
data['disease', 'rev_has_symptom', 'symptom'].edge_index = disease_to_symptom_edge_index

# ========== 加载模型 ==========
model = Symptom2DiseaseGNN(hidden_dim=128, out_dim=len(diseases))
model.load_state_dict(torch.load("gnn_model.pt", map_location="cpu"))
model.eval()

print("✅ GNN 模型加载完成！服务已准备就绪")

# ========== 预测 API ==========
@app.route('/api/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    input_symptoms = input_data.get('symptoms', [])

    if not input_symptoms:
        return jsonify([])

    # 映射症状 → 索引
    symptom_indices = []
    for sid, name in symptoms.items():
        if name in input_symptoms:
            symptom_indices.append(symptom_id_map[sid])

    if not symptom_indices:
        return jsonify([])

    # 推理
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        probs = torch.sigmoid(out)
        joint_pred = probs[symptom_indices].mean(dim=0)
        topk_vals, topk_idxs = torch.topk(joint_pred, 5)

    # 构造响应
    results = []
    for val, idx in zip(topk_vals.tolist(), topk_idxs.tolist()):
        for did, i in disease_id_map.items():
            if i == idx:
                results.append({
                    "name": diseases[did],
                    "probability": round(float(val), 4)
                })
                break

    return jsonify(results)

# ========== 启动服务 ==========
if __name__ == "__main__":
    app.run(debug=True)