# LangChain Integration

Integrate i2i's multi-model consensus verification into your LangChain pipelines.

## Installation

```bash
pip install i2i-mcip[langchain]
```

Or install LangChain separately:

```bash
pip install i2i-mcip langchain-core>=0.1.0
```

## Quick Start

```python
from i2i.integrations.langchain import I2IVerifier
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Create your LangChain components
llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_template("Answer: {question}")

# Add verification to your chain
chain = prompt | llm | I2IVerifier(min_confidence=0.8)

# Use normally - responses are now verified
result = chain.invoke({"question": "What is the capital of France?"})

print(result.verified)           # True
print(result.consensus_level)    # "HIGH"
print(result.content)            # "Paris is the capital of France..."
```

## Configuration Options

### I2IVerifier

The main Runnable component for adding verification to LCEL chains.

```python
from i2i.schema import ConsensusLevel

I2IVerifier(
    models: Optional[List[str]] = None,              # Models for consensus
    min_confidence: float = 0.7,                     # Confidence threshold (0-1)
    confidence_threshold: Optional[float] = None,   # Alias for min_confidence
    min_consensus_level: ConsensusLevel = MEDIUM,   # Minimum consensus level required
    task_category: Optional[str] = None,            # Override task detection
    task_aware: bool = True,                        # Enable task-aware routing
    protocol: Optional[AICP] = None,                # Pre-configured AICP instance
    raise_on_failure: bool = False,                 # Raise exception on verification failure
    include_verification_metadata: bool = True,     # Include detailed metadata in output
    statistical_mode: bool = False,                 # Use statistical consensus
    n_runs: int = 3,                                # Runs per model in statistical mode
    temperature: float = 0.7,                       # Temperature for statistical queries
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | `List[str]` | `None` | Models for consensus queries. Auto-selects if `None`. |
| `min_confidence` | `float` | `0.7` | Minimum calibrated confidence to pass verification. |
| `confidence_threshold` | `float` | `None` | Alias for `min_confidence`. |
| `min_consensus_level` | `ConsensusLevel` | `MEDIUM` | Minimum consensus level (HIGH > MEDIUM > LOW > NONE). |
| `task_category` | `str` | `None` | Override automatic task detection (`"factual"`, `"reasoning"`, `"creative"`). |
| `task_aware` | `bool` | `True` | Use task-aware routing (skips consensus for math/reasoning). |
| `protocol` | `AICP` | `None` | Pre-configured i2i protocol instance. |
| `raise_on_failure` | `bool` | `False` | Raise `VerificationError` when verification fails. |
| `include_verification_metadata` | `bool` | `True` | Include detailed verification metadata in output. |

Access configuration via the `config` property:

```python
verifier = I2IVerifier(confidence_threshold=0.9, min_consensus_level=ConsensusLevel.HIGH)
print(verifier.config.confidence_threshold)  # 0.9
print(verifier.config.min_consensus_level)   # ConsensusLevel.HIGH
```

### VerificationConfig

For advanced configuration:

```python
from i2i.integrations.langchain import VerificationConfig

config = VerificationConfig(
    models=["gpt-4", "claude-3-opus"],
    min_consensus_level="MEDIUM",
    confidence_threshold=0.7,
    task_aware=True,
    raise_on_failure=False,
    fallback_on_error=True,
    statistical_mode=False,
    n_runs=3,
    temperature=0.7,
)
```

## Output Model

`I2IVerifiedOutput` contains the verification result:

```python
@dataclass
class I2IVerifiedOutput:
    content: str                           # Original verified content
    verified: bool                         # Whether it passed verification
    consensus_level: str                   # HIGH, MEDIUM, LOW, NONE, CONTRADICTORY
    confidence_calibration: Optional[float] # Calibrated confidence 0.0-1.0
    task_category: Optional[str]           # Detected task type
    consensus_appropriate: Optional[bool]  # Whether consensus was appropriate
    models_queried: List[str]              # Models that participated
    original_metadata: Dict[str, Any]      # Preserved input metadata
```

## Usage Patterns

### Pattern 1: Basic LCEL Chain

```python
from i2i.integrations.langchain import I2IVerifier
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_template("What is {question}?")

chain = prompt | llm | I2IVerifier(min_confidence=0.8)

result = chain.invoke({"question": "the speed of light"})
print(f"Verified: {result.verified}")
print(f"Consensus: {result.consensus_level}")
```

### Pattern 2: Async Verification

```python
async def verify_response():
    verifier = I2IVerifier(
        models=["gpt-4", "claude-3-opus"],
        min_confidence=0.85,
        task_aware=True
    )

    result = await verifier.ainvoke("The Earth orbits the Sun")
    print(f"Verified: {result.verified}")
    print(f"Confidence: {result.confidence_calibration}")
```

### Pattern 3: Callback-Based Automatic Verification

```python
from i2i.integrations.langchain import I2IVerificationCallback
from i2i import ConsensusLevel

callback = I2IVerificationCallback(
    min_consensus_level=ConsensusLevel.HIGH,
    on_verification_failure="warn"  # "warn", "raise", or "ignore"
)

llm = ChatOpenAI(callbacks=[callback])
response = llm.invoke("Your prompt here")

# Access verification results
last = callback.get_last_verification()
history = callback.get_verification_history()
```

### Pattern 4: Verified Chain Wrapper

```python
from i2i.integrations.langchain import I2IVerifiedChain

base_chain = prompt | llm
verified_chain = I2IVerifiedChain(
    chain=base_chain,
    min_consensus_level=ConsensusLevel.HIGH
)

result = await verified_chain.ainvoke({"topic": "quantum computing"})
```

### Pattern 5: Helper Function

```python
from i2i.integrations.langchain import create_verified_chain

verified_chain = create_verified_chain(
    chain=prompt | llm,
    confidence_threshold=0.9,
    task_aware=True
)

result = await verified_chain.ainvoke({"question": "What is 2+2?"})
```

## RAG Verification

i2i is particularly useful for verifying RAG pipeline outputs to detect hallucinations.

### Basic RAG + Verification

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from i2i.integrations.langchain import I2IVerifier

# Setup RAG components
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(documents, embeddings)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-4")

prompt = ChatPromptTemplate.from_template("""
Answer based on context:
{context}

Question: {question}
""")

# RAG chain with verification
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | I2IVerifier(min_confidence=0.8)
)

result = rag_chain.invoke("What is our return policy?")
if not result.verified:
    print("Warning: Response may contain hallucination")
```

### Production RAG Pattern

```python
class ProductionVerifiedRAG:
    def __init__(
        self,
        retriever,
        llm,
        confidence_accept: float = 0.85,
        confidence_review: float = 0.60,
    ):
        self.retriever = retriever
        self.llm = llm
        self.confidence_accept = confidence_accept
        self.confidence_review = confidence_review
        self.verifier = I2IVerifier(task_aware=True)

    async def query(self, question: str) -> dict:
        # Retrieve context
        docs = await self.retriever.aget_relevant_documents(question)
        context = "\n".join(d.page_content for d in docs)

        # Generate response
        response = await self.llm.ainvoke(
            f"Context: {context}\n\nQuestion: {question}"
        )

        # Verify
        result = await self.verifier.ainvoke(response.content)

        # Categorize by confidence
        if result.confidence_calibration >= self.confidence_accept:
            status = "accepted"
        elif result.confidence_calibration >= self.confidence_review:
            status = "needs_review"
        else:
            status = "rejected"

        return {
            "answer": result.content,
            "status": status,
            "confidence": result.confidence_calibration,
            "consensus_level": result.consensus_level,
            "sources": [d.metadata.get("source") for d in docs],
        }
```

## Task-Aware Behaviour

i2i automatically detects task types and adjusts verification accordingly:

| Task Type | Consensus Behaviour | Reason |
|-----------|-------------------|--------|
| Factual | Full consensus | HIGH consensus = 97-100% accuracy |
| Verification | Full consensus | +6% improvement in detection |
| Mathematical | Skip consensus | Consensus degrades math by ~35% |
| Creative | Skip consensus | Consensus flattens diversity |

```python
# Automatic task detection
result = await verifier.ainvoke("What is the capital of France?")
print(result.task_category)          # "factual"
print(result.consensus_appropriate)  # True

result = await verifier.ainvoke("Calculate 5 * 3 + 2")
print(result.task_category)          # "reasoning"
print(result.consensus_appropriate)  # False
```

Override automatic detection:

```python
verifier = I2IVerifier(task_category="factual")  # Force factual handling
```

## Error Handling

### Fallback Mode (Default)

```python
verifier = I2IVerifier(fallback_on_error=True)  # Default
# If verification fails, returns original content with verified=False
```

### Strict Mode

```python
from i2i.integrations.langchain import VerificationError

verifier = I2IVerifier(raise_on_failure=True)

try:
    result = await verifier.ainvoke("Some content")
except VerificationError as e:
    print(f"Verification failed: {e}")
```

## Streaming Support

The verifier supports streaming, though verification happens after full content is received:

```python
async for chunk in verifier.astream("Content to verify"):
    # Single chunk with full verified result
    print(chunk.verified)
```

## Input Types

The verifier accepts multiple input formats:

```python
# String input
result = verifier.invoke("Plain text content")

# LangChain AIMessage
from langchain_core.messages import AIMessage
result = verifier.invoke(AIMessage(content="Message content"))

# LLMResult
from langchain_core.outputs import LLMResult
result = verifier.invoke(llm_result)

# Dict with 'content' key
result = verifier.invoke({"content": "Content here", "metadata": {}})
```

## API Reference

### Classes

#### I2IVerifier

Main Runnable for verification.

```python
class I2IVerifier(Runnable[Input, I2IVerifiedOutput]):
    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None
    ) -> I2IVerifiedOutput: ...

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None
    ) -> I2IVerifiedOutput: ...

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None
    ) -> Iterator[I2IVerifiedOutput]: ...

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None
    ) -> AsyncIterator[I2IVerifiedOutput]: ...
```

#### I2IVerificationCallback

Callback handler for automatic verification.

```python
class I2IVerificationCallback(BaseCallbackHandler):
    def __init__(
        self,
        min_consensus_level: ConsensusLevel = ConsensusLevel.MEDIUM,
        on_verification_failure: str = "warn",  # "warn", "raise", "ignore"
    ): ...

    def get_last_verification(self) -> Optional[I2IVerifiedOutput]: ...
    def get_verification_history(self) -> List[I2IVerifiedOutput]: ...
    def clear_history(self) -> None: ...
```

#### I2IVerifiedChain

Wrapper for adding verification to any chain.

```python
class I2IVerifiedChain:
    def __init__(
        self,
        chain: Runnable,
        min_consensus_level: ConsensusLevel = ConsensusLevel.MEDIUM,
        **kwargs
    ): ...

    def invoke(self, input: Any) -> I2IVerifiedOutput: ...
    async def ainvoke(self, input: Any) -> I2IVerifiedOutput: ...
```

### Functions

#### create_verified_chain

Helper to create a verified chain.

```python
def create_verified_chain(
    chain: Runnable,
    models: Optional[List[str]] = None,
    min_consensus_level: ConsensusLevel = ConsensusLevel.MEDIUM,
    confidence_threshold: float = 0.7,
    task_aware: bool = True,
    protocol: Optional[AICP] = None,
) -> Runnable: ...
```

## Requirements

- Python 3.9+
- i2i-mcip
- langchain-core >= 0.1.0
- At least 2 configured LLM providers for consensus

## Environment Variables

```bash
# Provider API keys (need at least 2)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Optional: Configure default models
I2I_CONSENSUS_MODEL_1=gpt-4
I2I_CONSENSUS_MODEL_2=claude-3-opus
```

## Confidence Calibration

Calibrated confidence scores based on empirical evaluation:

| Consensus Level | Confidence Score | Interpretation |
|-----------------|------------------|----------------|
| HIGH (≥85%) | 0.95 | Trust the answer |
| MEDIUM (60-84%) | 0.75 | Probably correct |
| LOW (30-59%) | 0.60 | Use with caution |
| NONE (<30%) | 0.50 | Likely hallucination |

## Troubleshooting

### ImportError: langchain-core not installed

```bash
pip install langchain-core>=0.1.0
# or
pip install i2i-mcip[langchain]
```

### Verification always returns verified=False

Check that you have at least 2 providers configured:

```bash
python -c "from i2i import AICP; print(AICP().registry.list_configured())"
```

### Slow verification

Consensus requires multiple API calls. For faster verification:
- Use faster models (e.g., `gpt-3.5-turbo`, `claude-3-haiku`)
- Reduce number of models
- Use `task_aware=True` to skip consensus when inappropriate

## See Also

- [i2i README](../../README.md) - Main documentation
- [RFC-MCIP](../../RFC-MCIP.md) - Protocol specification
- [Examples](../../examples/) - Jupyter notebooks with examples
