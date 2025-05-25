# Projeto DIO: Organização e Pesquisa de Documentos com IA - Simulação Detalhada

## 📖 Descrição do Projeto

Este projeto foi desenvolvido como parte do desafio "Aplicando Técnicas de Organização e Pesquisa de Documentos com Inteligência Artificial" da [Digital Innovation One (DIO)](https://web.dio.me/). O objetivo principal é aplicar técnicas de ingestão de dados e indexação utilizando ferramentas de inteligência artificial para organizar e permitir a pesquisa eficiente em um conjunto de documentos. Neste laboratório simulado, explorei o processo desde a preparação de artigos sobre Ética em Inteligência Artificial até a extração de conhecimento através de um sistema de perguntas e respostas.

## 🎯 Objetivos de Aprendizagem

Com a conclusão deste desafio (simulado), busquei desenvolver as seguintes habilidades:
* Aplicar conceitos de ingestão de dados e indexação com IA em um ambiente prático.
* Documentar processos técnicos de forma clara, estruturada e detalhada.
* Utilizar o GitHub como ferramenta para versionamento e compartilhamento de documentação técnica e projetos.
* Compreender como ferramentas de IA podem ser utilizadas para minerar e extrair conhecimento de grandes volumes de informação.

## 🛠️ Ferramentas e Tecnologias Utilizadas

* **Linguagem de Programação:** Python 3.10
* **Ambiente de Desenvolvimento:** Jupyter Notebook (executado via VS Code)
* **Bibliotecas Principais:**
    * **Langchain (v0.1.x):** Para orquestração do fluxo de RAG (Retrieval Augmented Generation).
        * `PyPDFLoader`: Para carregar documentos PDF.
        * `RecursiveCharacterTextSplitter`: Para dividir os textos em chunks menores.
        * `OpenAIEmbeddings`: Para gerar embeddings dos textos.
        * `FAISS`: Para criar e utilizar um índice vetorial local.
        * `RetrievalQA`: Para construir a cadeia de perguntas e respostas.
        * `OpenAI` (LLM): Para geração das respostas com base nos contextos recuperados.
    * **FAISS-CPU:** Biblioteca para busca de similaridade em vetores, utilizada como vector store local.
    * **OpenAI Python SDK:** Para interagir com os modelos da OpenAI (embeddings e LLM).
    * **Dotenv:** Para gerenciar chaves de API de forma segura.
* **Modelos de Linguagem (LLMs):**
    * `text-embedding-ada-002` (OpenAI): Para geração de embeddings.
    * `gpt-3.5-turbo-instruct` ou `gpt-3.5-turbo` (OpenAI): Para a geração de respostas.
* **Fonte de Dados:** 3 Artigos públicos em formato PDF sobre Ética em Inteligência Artificial.
    * *Exemplo de fontes (hipotéticas para esta simulação): "The Ethics of Artificial Intelligence" - Nick Bostrom & Eliezer Yudkowsky (Capítulo de livro), "Asilomar AI Principles" - Future of Life Institute, "Ethical Guidelines for Trustworthy AI" - European Commission.*
* **Versionamento:** Git e GitHub.

*Nota sobre API Keys: Para utilizar os modelos da OpenAI, foi necessário configurar uma API Key. Em um projeto real, é crucial proteger essa chave, por exemplo, utilizando arquivos `.env` e adicionando `.env` ao `.gitignore`.*

## 🚀 Etapas do Projeto

O projeto foi dividido nas seguintes etapas principais:

### 1. Ingestão de Conteúdo para IA

Nesta etapa, o foco foi preparar e carregar os artigos sobre ética em IA para que pudessem ser processados.

* **Descrição do Processo:**
    * **Coleta de Dados:** Foram selecionados 3 artigos em formato PDF que abordam diferentes aspectos da ética em Inteligência Artificial. Os arquivos foram salvos em um diretório local (`./documentos/`).
    * **Carregamento:** Utilizei a classe `PyPDFLoader` da Langchain para carregar o conteúdo de cada PDF. Cada página do PDF foi inicialmente carregada como um "Documento" Langchain.
    * **Divisão em Chunks (Chunking):** Para otimizar o processo de embedding e a busca de contexto, os textos carregados foram divididos em pedaços menores (chunks) utilizando `RecursiveCharacterTextSplitter`. Configurei um `chunk_size` de 1000 caracteres e um `chunk_overlap` de 150 caracteres. O overlap ajuda a manter o contexto entre chunks adjacentes.
* **Ferramentas/Comandos Utilizados (Exemplos de Código):**
    ```python
    # Exemplo de carregamento de um PDF
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader("./documentos/artigo_etica_ia_01.pdf")
    pages = loader.load()

    # Exemplo de divisão em chunks
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(pages) # 'pages' seria a junção de todos os documentos carregados
    ```
* **Insights e Desafios:**
    * **Qualidade do PDF:** Percebi que a qualidade do PDF original é crucial. PDFs que são imagens escaneadas exigiriam OCR (Optical Character Recognition) antes do carregamento, o que não foi o foco aqui, mas é um ponto de atenção.
    * **Ajuste de Chunking:** Experimentar com `chunk_size` e `chunk_overlap` foi importante. Chunks muito pequenos podem perder contexto, enquanto chunks muito grandes podem ser menos eficientes para a busca de similaridade e podem exceder limites de tokens do modelo de embedding ou LLM.
    * **Metadados:** Para esta simulação, não adicionei metadados complexos aos chunks, mas percebi que em projetos maiores, adicionar a fonte (nome do arquivo, página) a cada chunk seria muito útil para rastreabilidade.

### 2. Criação de Índices Inteligentes

Com os dados ingeridos e divididos, o próximo passo foi gerar "embeddings" e criar um índice vetorial para busca semântica.

* **Descrição do Processo:**
    * **Geração de Embeddings:** Utilizei `OpenAIEmbeddings` (com o modelo `text-embedding-ada-002`) para converter cada chunk de texto em um vetor numérico (embedding). Esses vetores capturam o significado semântico do texto.
    * **Criação do Vector Store:** Os embeddings gerados, juntamente com os chunks de texto originais, foram armazenados em um índice vetorial local usando `FAISS`. O FAISS permite buscas rápidas por similaridade entre um vetor de consulta e os vetores armazenados.
    * **Persistência (Opcional, mas recomendado):** Para evitar recriar o índice a cada execução, o índice FAISS pode ser salvo localmente (`vectorstore.save_local("faiss_index_etica_ia")`) e carregado posteriormente (`FAISS.load_local(...)`).
* **Ferramentas/Comandos Utilizados (Exemplos de Código):**
    ```python
    # Configurando API Key (exemplo, melhor usar dotenv)
    # import os
    # os.environ["OPENAI_API_KEY"] = "SUA_CHAVE_API_AQUI"

    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Criando o índice FAISS a partir dos documentos (chunks) e seus embeddings
    # 'docs' são os chunks da etapa anterior
    vectorstore = FAISS.from_documents(docs, embeddings_model)

    # Salvando o índice (opcional)
    # vectorstore.save_local("faiss_index_etica_ia")
    ```
* **Insights e Desafios:**
    * **Custo de Embeddings:** A geração de embeddings com modelos como o da OpenAI tem um custo associado (embora pequeno para este volume). Em projetos maiores, otimizar o número de chunks e escolher modelos de embedding eficientes é crucial.
    * **Escolha do Vector Store:** FAISS é excelente para começar e para uso local devido à sua simplicidade. Para aplicações em produção ou com volumes de dados massivos, outras soluções como Pinecone, Weaviate ou ChromaDB poderiam ser consideradas, oferecendo mais funcionalidades e escalabilidade.
    * **"Magia" dos Embeddings:** No início, o conceito de embeddings pode parecer abstrato, mas ao ver a busca por similaridade funcionando, fica claro o poder de representar texto numericamente de forma semântica.

### 3. Exploração Prática dos Dados Organizados

A etapa final consistiu em utilizar o índice criado para realizar buscas e obter respostas a perguntas sobre os documentos de ética em IA.

* **Descrição do Processo:**
    * **Configuração do Retriever:** O índice FAISS foi configurado como um `retriever` na Langchain. O retriever é responsável por buscar os chunks mais relevantes para uma dada consulta. Configurei para retornar os top 3 chunks (`k=3`).
    * **Cadeia de QA (Question Answering):** Utilizei a cadeia `RetrievalQA` da Langchain. Essa cadeia combina o retriever com um modelo de linguagem (LLM, no caso `gpt-3.5-turbo`). O fluxo é:
        1. A pergunta do usuário é usada para buscar chunks relevantes no vector store (via retriever).
        2. Os chunks recuperados (contexto) e a pergunta original são passados para o LLM.
        3. O LLM gera uma resposta com base no contexto fornecido.
    * **Realização de Perguntas:** Formulei perguntas específicas sobre o conteúdo dos artigos de ética em IA.
* **Ferramentas/Comandos Utilizados (Exemplos de Código):**
    ```python
    from langchain_openai import OpenAI
    from langchain.chains import RetrievalQA

    # Carregando o LLM
    llm = OpenAI(temperature=0.2, model_name="gpt-3.5-turbo-instruct") # ou ChatOpenAI para modelos de chat

    # Configurando o retriever a partir do vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Criando a cadeia de QA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" simplesmente "enfia" todos os chunks recuperados no prompt
        retriever=retriever,
        return_source_documents=True # Opcional, para ver quais chunks foram usados
    )

    # Exemplo de pergunta
    pergunta = "Quais são os principais princípios éticos para uma IA confiável segundo a Comissão Europeia?"
    resposta = qa_chain.invoke({"query": pergunta})

    print("Pergunta:", pergunta)
    print("Resposta:", resposta["result"])
    # print("Documentos Fonte:", resposta["source_documents"]) # Se return_source_documents=True
    ```
* **Resultados e Insights:**
    * **Qualidade das Respostas:** As respostas geradas pelo LLM foram, em geral, relevantes e baseadas nos trechos recuperados dos PDFs. Por exemplo, ao perguntar sobre "princípios da IA confiável da Comissão Europeia", o sistema conseguiu extrair e resumir os pontos relevantes do documento correspondente.
    * **Importância do Contexto:** A qualidade da resposta depende diretamente da qualidade e relevância dos chunks recuperados. Se o retriever não encontrar o contexto correto, o LLM não terá base para responder adequadamente ou poderá "alucinar".
    * **Tipos de Cadeia (`chain_type`):** Experimentei inicialmente com `chain_type="stuff"`, que é simples mas pode ter limitações com contextos muito longos. Para documentos maiores, outros tipos como `map_reduce` ou `refine` seriam mais apropriados.
    * **Engenharia de Prompt (Implícita):** Embora não tenha customizado extensivamente o prompt da cadeia `RetrievalQA`, percebi que a forma como a pergunta é feita influencia a busca e a resposta. Perguntas mais específicas tendem a gerar resultados melhores.
    * **Limitações:** O sistema só "sabe" o que está nos documentos fornecidos. Perguntas fora desse escopo não serão respondidas corretamente.

## 🌟 Conclusão

Este laboratório simulado foi uma excelente oportunidade para aplicar na prática os conceitos de ingestão, indexação e busca de informações com IA, utilizando a abordagem RAG com Langchain. A capacidade de transformar documentos textuais em uma base de conhecimento pesquisável semanticamente é poderosa.

Principais aprendizados:
* A importância de cada etapa: uma boa ingestão e chunking levam a melhores embeddings, que resultam em uma recuperação de contexto mais precisa e, consequentemente, respostas mais assertivas do LLM.
* O ecossistema Langchain simplifica enormemente a construção de aplicações complexas com LLMs, abstraindo muitas das complexidades.
* Ferramentas como FAISS tornam a criação de índices vetoriais acessível mesmo para desenvolvimento local.
* A necessidade de experimentação (tamanho dos chunks, modelos de embedding/LLM, tipos de cadeia) é fundamental para otimizar o desempenho para um caso de uso específico.

Este projeto reforçou minha compreensão sobre como as ferramentas de IA podem ser utilizadas para minerar e extrair conhecimento valioso de grandes volumes de informação não estruturada.

## ✨ Agradecimentos

Agradeço à DIO pela oportunidade de realizar este desafio e aprender mais sobre o fascinante mundo da Inteligência Artificial aplicada.
