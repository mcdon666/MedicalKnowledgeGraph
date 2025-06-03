const express = require('express')
const cors = require('cors')
const neo4j = require('neo4j-driver')

const app = express()
app.use(cors())

const driver = neo4j.driver(
  'bolt://localhost:7687',  // 替换成你的Neo4j地址
  neo4j.auth.basic('neo4j', '12345678')
)
const session = driver.session()

app.get('/api/departments', async (req, res) => {
  try {
    const result = await session.run('MATCH (d:Department) RETURN d.name AS name')
    const departments = result.records.map(r => r.get('name'))
    res.json(departments)
  } catch (error) {
    console.error(error)
    res.status(500).send('Server error')
  }
})

app.listen(4000, () => {
  console.log('Server started on http://localhost:4000')
})
