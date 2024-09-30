
# Agent RAG local avec LLaMA 3

Ce projet présente un agent RAG (Retrieval-Augmented Generation) local utilisant LLaMA 3, combinant plusieurs idées de papers récents sur RAG :

- **Routage** : Le routage adaptatif RAG, qui redirige les questions vers différentes méthodes de récupération.
- **Fallback** : L'agent RAG correctif, qui effectue une recherche sur le web si les documents ne sont pas pertinents.
- **Auto-correction** : Self-RAG, qui corrige les réponses contenant des hallucinations ou qui ne répondent pas à la question.

## Modèles locaux

### Embedding :
- **GPT4All Embeddings** : Utilisation de GPT4All pour les embeddings des documents.

### Modèles LLM :
- **Ollama et Llama 3.2** : Utilisation du modèle `llama3.2:3b-instruct-fp16` pour l'inférence locale.

### Recherche Web :
- **Tavily** : Moteur de recherche optimisé pour les LLMs et RAG, intégré pour les recherches web.

## Instructions d'utilisation

### Installation des dépendances
Exécutez la commande suivante pour installer les bibliothèques requises :

```bash
%pip install --quiet -U langchain langchain_community tiktoken langchain-nomic "nomic[local]" langchain-ollama scikit-learn langgraph tavily-python bs4
```

### Configuration des clés API

Vous devez définir vos clés API pour Tavily et LangSmith. Utilisez les variables d'environnement suivantes dans le notebook :

```python
_set_env("TAVILY_API_KEY")
_set_env("LANGCHAIN_API_KEY")
```

### Chargement des documents

Voici un exemple de chargement de documents à partir d'URLs spécifiques et de création d'un vecteur de documents à utiliser dans la base de données vectorielle :

```python
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Chargement des documents
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Découpage des documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
)
doc_splits = text_splitter.split_documents(docs_list)

# Ajout à la base de données vectorielle
vectorstore = SKLearnVectorStore.from_documents(
    documents=doc_splits,
    embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
)

# Création d'un récupérateur (retriever)
retriever = vectorstore.as_retriever(k=3)
```

### Routage des questions

L'agent utilise un système de routage intelligent pour décider si une question doit être envoyée à la base de données vectorielle ou à une recherche sur le web.

Voici un exemple de routage de questions :

```python
test_web_search = llm_json_mode.invoke([SystemMessage(content=router_instructions)] + [HumanMessage(content="Qui est favori pour remporter la finale de la NFC en 2024 ?")])
test_vector_store = llm_json_mode.invoke([SystemMessage(content=router_instructions)] + [HumanMessage(content="Quels sont les types de mémoire d'agent ?")])
```

### Grader pour évaluer la pertinence des documents

L'agent évalue automatiquement la pertinence des documents récupérés pour une question donnée :

```python
doc_grader_prompt = """Voici le document récupéré : {document}. Voici la question utilisateur : {question}. Évaluez si le document contient des informations pertinentes."""
```

### Génération et post-traitement

Le modèle génère une réponse concise basée sur les documents récupérés, avec un prompt structuré pour assurer la clarté et la concision.

```python
rag_prompt = """Vous êtes un assistant pour les tâches de question-réponse. Voici le contexte : {context}. Répondez à la question : {question}. Réponse :"""
```

### Grader de la génération

Un Grader vérifie que la génération ne contient pas d'hallucinations et est bien fondée sur les faits présentés dans les documents :

```python
hallucination_grader_prompt = """Voici les faits : {documents}. Voici la réponse de l'élève : {generation}. Scorez 'oui' si la réponse est bien fondée sur les faits."""
```

## Graphique du flux de travail

L'agent suit un graphe d'état définissant le flux de travail entre les étapes de récupération, génération, et évaluation des réponses. Ce graphe est construit avec LangGraph, et il est visible dans les traces de l'exécution.

Pour un exemple en temps réel, référez-vous aux liens suivants :
- [Trace 1](https://smith.langchain.com/public/1e01baea-53e9-4341-a6d1-b1614a800a97/r)
- [Trace 2](https://smith.langchain.com/public/acdfa49d-aa11-48fb-9d9c-13a687ff311f/r)
