The code for the paper **"PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths"**.
## Install
```bash
cd PathRAG
pip install -e .
```
This package requires optional dependencies when using certain backends. To enable
the default Kuzu graph storage, install the `kuzu` package:
```bash
pip install kuzu
```
## Quick Start
* You can quickly experience this project in the `examples/v1_example.py` file.
* Set OpenAI API key in environment if using OpenAI models: `api_key="sk-...".` in the `examples/v1_example.py` and `llm.py` file
* Prepare your retrieval document "text.txt".
* Use the following Python snippet in the `examples/v1_example.py` file to initialize PathRAG and perform queries.
  
```python
import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import gpt_4o_mini_complete

WORKING_DIR = "./your_working_dir"
api_key="your_api_key"
os.environ["OPENAI_API_KEY"] = api_key
base_url="https://api.openai.com/v1"
os.environ["OPENAI_API_BASE"]=base_url


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,  
)

data_file="./text.txt"
question="your_question"
with open(data_file) as f:
    rag.insert(f.read())

print(rag.query(question, param=QueryParam(mode="hybrid")))
```

## API Reference

`PathRAG` exposes a small set of high-level methods:

- `insert(text_or_list)` – chunk and index new documents
- `insert_custom_kg(graph_dict)` – load an existing knowledge graph
- `query(question, param=QueryParam())` – retrieve context and generate an answer
- `delete_by_entity(name)` – remove all data related to an entity

Synchronous calls have async counterparts (`ainsert`/`aquery`).

By default `PathRAG` uses `KuzuGraphStorage` to persist and query the knowledge graph.
You can switch to the in-memory `NetworkXStorage` by passing `graph_storage="NetworkXStorage"`.

## Command Line Interface

PathRAG also provides a simple CLI for inserting documents, querying and deleting entities. Run it using the module syntax:

```bash
python -m PathRAG.cli insert --file text.txt --working-dir ./cache
python -m PathRAG.cli query --question "What is PathRAG?" --working-dir ./cache
python -m PathRAG.cli delete --entity "SOME_ENTITY" --working-dir ./cache
```

The CLI shares the same defaults as the Python API and reads your OpenAI credentials from the environment.

## Dagster Assets with S3

PathRAG can be integrated into a Dagster pipeline. The `examples/dagster_assets.py` file
defines three assets that index a text document and run a query while storing
asset materializations on S3 using `S3PickleIOManager`.

```bash
pip install dagster dagster-aws
dagster dev -m examples.dagster_assets
```
## Parameter modification
You can adjust the relevant parameters in the `base.py` and `operate.py` files.

## Batch Insert
```python
import os
folder_path = "your_folder_path"  

txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
for file_name in txt_files:
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "r", encoding="utf-8") as file:
        rag.insert(file.read())
```

## Cite
Please cite our paper if you use this code in your own work:
```python
@article{chen2025pathrag,
  title={PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths},
  author={Chen, Boyu and Guo, Zirui and Yang, Zidan and Chen, Yuluo and Chen, Junze and Liu, Zhenghao and Shi, Chuan and Yang, Cheng},
  journal={arXiv preprint arXiv:2502.14902},
  year={2025}
}
```
