import nest_asyncio
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
import os
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_parse import LlamaParse
from llama_index.core.tools import QueryEngineTool
from llama_index.core.workflow import (
    step,
    Event,
    Context,
    StartEvent,
    StopEvent,
    Workflow,
)
from llama_index.core.agent import FunctionCallingAgent

nest_asyncio.apply()

_ = load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini")

Settings.embed_model = OpenAIEmbedding(
    model_name="text-embedding-3-small"
)

# Create New Index from Document

DATA_DIR = "./doc"
PERSIST_DIR = "./storage"

if os.path.exists(PERSIST_DIR):
    print("Loading existing index...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
else:
    print("Creating new index...")
    file_path = "./doc/Newwhitepaper_Agents2.pdf"
    documents = LlamaParse(result_type="markdown").load_data(file_path)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

# Run a Query Against the Index
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query(
    "What are agents?"
)
print(response)

# create engine query tool
agent_rag_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="google_agents_tool",  # Changed to use underscores instead of spaces
    description="A RAG engine with extremely detailed information about Google's perspective on AI agents.",
)

# Build the Agent
class OutlineEvent(Event):
    outline: str

class QuestionEvent(Event):
    question: str

class AnswerEvent(Event):
    question: str
    answer: str

class ReviewEvent(Event):
    report: str

class ProgressEvent(Event):
    progress: str

class DocumentResearchAgent(Workflow):
    # get the initial request and create an outline of the blog post knowing nothing about the topic
    
    @step()
    async def formulate_plan(
        self, ctx: Context, ev: StartEvent
    ) -> OutlineEvent:
        query = ev.query
        await ctx.set("original_query", query)
        await ctx.set("tools", ev.tools)

        prompt = f"""You are an expert at writing blog posts. You have been given a topic to write
        a blog post about. Plan an outline for the blog post; it should be detailed and specific.
        Another agent will formulate questions to find the facts necessary to fulfill the outline.
        The topic is: {query}"""

        response = await Settings.llm.acomplete(prompt)

        ctx.write_event_to_stream(
            ProgressEvent(progress="Outline:\n" + str(response))
        )

        return OutlineEvent(outline=str(response))
    
    # formulate some questions based on the outline
    @step()
    async def formulate_questions(
        self, ctx: Context, ev: OutlineEvent
    ) -> QuestionEvent:
        outline = ev.outline
        await ctx.set("outline", outline)

        prompt = f"""You are an expert at formulating research questions. You have been given an outline
        for a blog post. Formulate a series of simple questions that will get you the facts necessary
        to fulfill the outline. You cannot assume any existing knowledge; you must ask at least one
        question for every bullet point in the outline. Avoid complex or multi-part questions; break
        them down into a series of simple questions. Your output should be a list of questions, each
        on a new line. Do not include headers or categories or any preamble or explanation; just a
        list of questions. For speed of response, limit yourself to 8 questions. The outline is: {outline}"""

        response = await Settings.llm.acomplete(prompt)

        questions = str(response).split("\n")
        questions = [x for x in questions if x]

        ctx.write_event_to_stream(
            ProgressEvent(
                progress="Formulated questions:\n" + "\n".join(questions)
            )
        )

        await ctx.set("num_questions", len(questions))

        ctx.write_event_to_stream(
            ProgressEvent(progress="Questions:\n" + "\n".join(questions))
        )

        for question in questions:
            ctx.send_event(QuestionEvent(question=question))
    
    # answer each question in turn
    @step()
    async def answer_question(
        self, ctx: Context, ev: QuestionEvent
    ) -> AnswerEvent:
        question = ev.question
        if (
            not question
            or question.isspace()
            or question == ""
            or question is None
        ):
            ctx.write_event_to_stream(
                ProgressEvent(progress=f"Skipping empty question.")
            )  # Log skipping empty question
            return None
        agent = FunctionCallingAgent.from_tools(
            await ctx.get("tools"),
            verbose=True,
        )
        response = await agent.aquery(question)

        ctx.write_event_to_stream(
            ProgressEvent(
                progress=f"To question '{question}' the agent answered: {response}"
            )
        )

        return AnswerEvent(question=question, answer=str(response))
    
    # given all the answers to all the questions and the outline, write the blog poost
    @step()
    async def write_report(self, ctx: Context, ev: AnswerEvent) -> ReviewEvent:
        # wait until we receive as many answers as there are questions
        num_questions = await ctx.get("num_questions")
        results = ctx.collect_events(ev, [AnswerEvent] * num_questions)
        if results is None:
            return None

        # maintain a list of all questions and answers no matter how many times this step is called
        try:
            previous_questions = await ctx.get("previous_questions")
        except:
            previous_questions = []
        previous_questions.extend(results)
        await ctx.set("previous_questions", previous_questions)

        prompt = f"""You are an expert at writing blog posts. You are given an outline of a blog post
        and a series of questions and answers that should provide all the data you need to write the
        blog post. Compose the blog post according to the outline, using only the data given in the
        answers. The outline is in <outline> and the questions and answers are in <questions> and
        <answers>.
        <outline>{await ctx.get('outline')}</outline>"""

        for result in previous_questions:
            prompt += f"<question>{result.question}</question>\n<answer>{result.answer}</answer>\n"

        ctx.write_event_to_stream(
            ProgressEvent(progress="Writing report with prompt:\n" + prompt)
        )

        report = await Settings.llm.acomplete(prompt)

        return ReviewEvent(report=str(report))
    
    # review the report. If it still needs work, formulate some more questions.
    @step
    async def review_report(
        self, ctx: Context, ev: ReviewEvent
    ) -> StopEvent | QuestionEvent:
        # we re-review a maximum of 3 times
        try:
            num_reviews = await ctx.get("num_reviews")
        except:
            num_reviews = 1
        num_reviews += 1
        await ctx.set("num_reviews", num_reviews)

        report = ev.report

        prompt = f"""You are an expert reviewer of blog posts. You are given an original query,
        and a blog post that was written to satisfy that query. Review the blog post and determine
        if it adequately answers the query and contains enough detail. If it doesn't, come up with
        a set of questions that will get you the facts necessary to expand the blog post. Another
        agent will answer those questions. Your response should just be a list of questions, one
        per line, without any preamble or explanation. For speed, generate a maximum of 4 questions.
        The original query is: '{await ctx.get('original_query')}'.
        The blog post is: <blogpost>{report}</blogpost>.
        If the blog post is fine, return just the string 'OKAY'."""

        response = await Settings.llm.acomplete(prompt)

        if response == "OKAY" or await ctx.get("num_reviews") >= 3:
            ctx.write_event_to_stream(
                ProgressEvent(progress="Blog post is fine")
            )
            return StopEvent(result=report)
        else:
            questions = str(response).split("\n")
            await ctx.set("num_questions", len(questions))
            ctx.write_event_to_stream(
                ProgressEvent(progress="Formulated some more questions")
            )
            for question in questions:
                ctx.send_event(QuestionEvent(question=question))

# test the agent
async def main():
    agent = DocumentResearchAgent(timeout=600, verbose=True)

    handler = agent.run(
        query="Tell me about Googles perspective on AI Agents and their plan around that? ",
        tools=[agent_rag_tool],
    )

    async for ev in handler.stream_events():
        if isinstance(ev, ProgressEvent):
            print(ev.progress)
    final_result = await handler
    print("------- Blog post ----------\n", final_result)

import asyncio
asyncio.run(main())