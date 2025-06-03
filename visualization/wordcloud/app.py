from flask import Flask, request, jsonify
from neo4j import GraphDatabase
from flask_cors import CORS

from question_classifier import *
from question_parser import *
from answer_search import *


app = Flask(__name__)
CORS(app)

# 替换为你自己的 Neo4j 地址和密码
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))

class ChatBotGraph:
    def __init__(self):
        self.classifier = QuestionClassifier()
        self.parser = QuestionPaser()
        self.searcher = AnswerSearcher()

    def chat_main(self, sent):
        default_answer = '您好，本问答内容仅供参考，如有不适请及时就医。若未能回答请及时求助专业医生咨询。'
        res_classify = self.classifier.classify(sent)
        if not res_classify:
            return default_answer
        res_sql = self.parser.parser_main(res_classify)
        final_answers = self.searcher.search_main(res_sql)
        return '\n'.join(final_answers) if final_answers else default_answer

chatbot = ChatBotGraph()

# ========== 1. 疾病关联词云数据接口 ==========
@app.route("/api/related_diseases")
def related_diseases():
    name = request.args.get("name", "肝癌")
    query = '''
    MATCH (d1:Disease {name: $name})-[:acompany_with]-(d2:Disease)
    RETURN d2.name AS name, COUNT(*) AS weight
    ORDER BY weight DESC
    '''
    with driver.session() as session:
        result = session.run(query, name=name)
        return jsonify([{"name": row["name"], "value": row["weight"]} for row in result])


# ========== 2. 疾病图谱结构数据接口 ==========
@app.route("/api/graph")
def graph():
    name = request.args.get("name", "肝癌")
    query = '''
    MATCH (d1:Disease {name: $name})-[r:acompany_with]-(d2:Disease)
    RETURN d1, d2, r
    '''
    with driver.session() as session:
        result = session.run(query, name=name)
        nodes = {}
        edges = []

        for record in result:
            d1 = record["d1"]
            d2 = record["d2"]
            r = record["r"]

            for node in [d1, d2]:
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
                    "source": str(d1.id),
                    "target": str(d2.id),
                    "label": r.type
                }
            })

        return jsonify({
            "nodes": list(nodes.values()),
            "edges": edges
        })


# ========== 3. 医学问答接口 ==========
@app.route('/api/chat', methods=['POST'])
def chat_api():
    user_input = request.json.get('question')
    answer = chatbot.chat_main(user_input)
    return jsonify({"answer": answer})


# ========== 4. 启动服务 ==========
if __name__ == "__main__":
    app.run(debug=True)