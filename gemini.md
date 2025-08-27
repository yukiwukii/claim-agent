# AI-Powered Travel Insurance Claims Agent
This project was started for the Zurich AI Hyperchallenge. We are looking to formalize our findings in hopes of refining the project and to potentially use it as a tech demo or even implement it as a product for DiscoverMarket. 

This document is an attempt at documenting everything that went into the projects: explaining all decisions, architecture, possible ideas for production and any other relevant detail that is worth storing. Although I intend to write this to be as humanly-readable and understandable as possible, the sheer length and information density of the project may make that more difficult to those unaware of the project, our process and the libraries we decided to use. I recommend for these readers to use AI tools to get the big picture ideas, and then follow that up with reading the parts of interest.

## Libraries Used
Zurich was not very clear in their requirements and expectations of the end solution--I don't think even they properly understood what they were looking for. Naturally, we were quite lost on where to start, but we obviously needed a library or some sort of framework for agent-building. The two main libraries used for this task are `langchain` and `n8n`. We began the project with LangChain because it lends itself for more modularity and fine-grained control. However, when the hackathon deadline was pushed back, we decided that even LangChain did not give us the flow control we needed. Therefore, we migrated to `langgraph`, a framework that provides greater control over agent behavior through stateful directed graphs. In LangGraph, nodes represent functions or agents, edges define execution flow, and state persists and can be modified throughout the workflow. This architecture enables us to build complex yet *explainable* systems with cyclical workflows, conditional branching, multi-agent collaboration, and fine-tuned state management. The explainability aspect of LangGraph is very attractive for this project because in the world of insurance, actions and decisions are subject to a lot of scrutiny, so we must be able to have a good explanation for each action the agent takes. This was not fully possible in the way we had designed our LangChain solution as there was an LLM that decided what to do at each step. The new LangGraph architecture uses agents and LLMs at the node level, but not at the edge level. In other words, we define the graph's flow manually via fixed checks of our state object at predetermined stages of the graph instead of letting an LLM jump around the graph as it wishes. 

The heart of the project is `langgraph`. It handles the graph, calls to LLM and some asynchronous tasks (parallelized agent workflows). LangGraph works in tandem with `pydantic`, a library for data validation. In this case, we are using it to structure our graph state object and get structured output from LLMs. I believe LangGraph depends on Pydantic, but either way, Pydantic is really important because it enforces typing and adds new capabilities to regular dictionaries/jsons. We define a `BaseModel` with named keys and their respective types. We can then prompt LLMs with information and this `BaseModel` and get back type-validated structured output. This is important for ensuring that the state object is usable and that we will not run into any typing issues due to bad LLM output. 

Some specific nodes in the graph may have their own dependencies (e.g. `requests` or equivalent to interact with APIs like FlightRadar24). 


## Preliminary Assumptions
This project depends on the following assumptions:

1. There are four possible claim types (architecturally this is quite modular, so we can add/edit/remove these depending on the need):
     1. Travel Delay (requires: claim form, passport, boarding pass/itinerary, delay notification, receipt)
     2. Baggage Delay (requires: claim form, passport, boarding pass/itinerary, Property Irregularity Report, 
     3. Pre-travel Trip Cancellation (requires: claim form, passport, boarding pass/itinerary, proof of payment + invoices, proof of cancellation, cancellation documentation)
     4. Post-travel Trip Interruption (requires: claim form, passport, boarding pass/itinerary, proof of reason for interruption, proof of unused services due to interruption, additional expenses)
2. The user's claim form is the 'ground truth'. Of course, this is verified with our customer database to ensure that the person and their policy number are both legitimate, but the information in the claim form (claimed expenses, claim type, etc) are the ones we fully trust. Other documents must support the claim form, not the other way around. 

## State Object
The `state` object is a key feature in `LangGraph` and part of the reason why this framework in specific is so powerful. The main idea is that you create a Python class with the following header: `class State(TypedDict)` (requires the `typing` import). This then lets you define named keys and their corresponding value types. Primitive types like `int`, `str`, `bool` obviously work, but the real power comes from the `typing` and `typing_extensions` imports, which let us add `List`, `Annotated`, `Any`, and `Optional`, among other types. I will not define the actual `state` object here because this really depends on the implementation, but here is an example of what we can do with it:

```python
class State(TypedDict):
    session_id: str
    claim_id: Optional[str]
    
    # Conversation and messaging
    messages: List[Message]
    current_message: Optional[str]
    response_message: Optional[str]
    
    # Unified claim information
    claim_data: Dict[str, Any]  # All extracted info in one place
    documents: List[Document]
    
    # Processing state
    stage: Literal["intake", "processing", "decision", "review", "completed", "error"]
    context: Dict[str, Any]  # LLM's working memory and routing context
    routing_history: List[str]  # Track routing decisions to detect loops
    
    # Results and decisions
    eligibility: Optional[Dict[str, Any]]
    payout: Optional[Dict[str, Any]]
    recommendation: Optional[Dict[str, Any]]
    final_decision: Optional[Literal["APPROVED", "DENIED", "MANUAL_REVIEW"]]
    flight_verification: Optional[FlightVerification]  # Flight verification results
    customer_verification: Optional[Dict[str, Any]]  # Customer verification results
    fraud_assessment: Optional[Dict[str, Any]]  # Fraud/vigilance assessment
    
    # Metadata
    created_at: str
    updated_at: str
    processing_time_ms: Optional[int]
    
    # Error handling
    error: Optional[str]
    error_details: Optional[Dict[str, Any]]
    
    # Orchestrator fields
    checklist: ProcessingChecklist
    processing_path: Literal["document_priority", "claim_form_priority", "standard"]
    orchestrator_notes: List[str]
    last_action: str
    next_node: str  # For orchestrator routing
```

In practice, it would be smart to have an overall state, and give each subagent its own subset of the state. We can add other `TypedDict`s or other classes within the `State` class, making it extremely modular and quite frankly, very difficult to get right. I think more than half of the battle is making this main `State` class. Once you have that properly defined, making functions around this class becomes quite natural and just a task of reading specific inputs from said state and writing outputs back to it. I recommend using AI coding agents to get a decent draft of what the state should look like and to very carefully revise it to remove any unnecessary/redundant fields. I think you really have to approach building the state class understanding that *this* is the agent you are building. We are essentially constructing what the agent will be able to know and act on here, so it is extremely important to get right. I believe that the best way to construct a good `State` class is to first begin by sketching out the graph, including node names, edges, and a brief outline of what is needed for each node (like below!). As the graph is actually implemented, the `State` can be refined alongside it. For instance, boolean flags can be added as conditional nodes/edges are made, fields containing important information from uploads can be edited as we work through the document analysis stage, etc. 

A difficulty I haven't fully explored is how to manage both a state and a database containing similar information. My intuition tells me that the state should store mostly information that is relevant to the graph (e.g. boolean flags, information that is not fully processed yet, etc), and the database should store results (OCRed text, reconciliation results, structured output, etc). However, a natural question that follows from this is: what should we do with results that are necessary for other steps? For example, it is clear to me that we need to store OCR, document class and structured output for each document, but should we only store that information in the database and make read calls whenever we need it, or is it wise to store some of that information in the state object? I think the answer lies somewhere in between: maybe 'heavier' fields like OCR text can be stored in the database, while information like document class, and important structured output can be stored at the state level. Another possible structure is that the state object acts as storage for pointers to the specific fields in the database which store the information we are looking for. This lends itself quite nicely for even more modular design, but would require a much larger database. 


## Architecture
We have gone through many graph architecture designs in the process of building this solution. In this section I will focus on the latest design from 20250724 (see below). I recommend readers to open the graph on a separate tab to follow along. 
![design](20250731.svg)


### Stage 1
Stage 1 consists of preliminary operations. It is triggered by document upload. The main idea here is gathering all necessary documents to process the claim, verify the customer's identity, and ensure that our OCR and structured output give us all of the required information to begin our document analysis and settle the claim. Stage 1 sees a more tree-like structure that uses more conditional edges to keep track of what our next steps are. 

#### OCR
When the document upload API is triggered, we initiate OCR for that document. This is done by an LLM call to `gemini-2.5-flash`, but at the time of reading this, it is possible that there are better LLMs out there that accomplish this task. For example, `olmOCR` in May 2025 was quite a good model, but it was clear by the end of June that Gemini was a better option (due to its easier implementation and quality consistency). The output OCR is stored in our state object (in production, it should be stored in a database). This node then routes to "Conditional Node: Is claim form in state?"

#### Conditional Node: Is claim form in state?
We check if the claim form is in our state. When we upload our first document, this is false. Even if the first document actually is the claim form, so far we have not extracted it. We have its text content, not its information. If we do not have the claim form, then we route to the "Binary Classifier". Otherwise, we route to the "Multi-class Classifier". 

#### Binary Classifier
This node uses LLM, currently `gemini-2.5-flash`, to check if the OCR content is the claim form or not. If it is the claim form, then we route to the "Extract Claim Form" node. Otherwise, we route to the Add to Queue node.

#### Add to Queue
This node is helpful in ensuring we do not have to re-OCR documents that have already been OCRed. The idea is that as long as we do not have the claim form, we keep a queue of all of the uploaded documents that are *not* the claim form. Once we have confirmed that we have the claim form, we can parallelize the Multi-Class Classification and Structured Output process for these queued documents--this is just context for a different node, I am trying to indicate why this queue is useful. 
After adding the document to the queue, we are free to END our execution of the graph. I will reclarify here that ENDing the graph does not mean that the agent's job is done. In this case, far from it. ENDing the graph at this point lets us stall until the next document(s) are uploaded. After all, we have nothing left to do at this point. We are still waiting to receive the claim form, and potentially other documents. The rest of the graph logic can only resume once we have those uploads.

#### Extract Claim Form
In the case where the binary classifier outputed that the input text is the claim form, then we must extract its contents. We use another LLM call with a `Pydantic` schema to specify the structured output we want (claim type, customer name, policy number, among other relevant fields). After completion, we route to the "Verify Customer" node.

#### Verify Customer
This node verifies two important things: first, as the node name suggests, that the customer is a real person in our database. We crosscheck the customer name and policy number with our records. In practice, we may want to use some sort of strict fuzzy matching to eliminate any possible OCR errors. The second check is whether we have all necessary fields of our `Pydantic` structure filled out with at least some sort of information. Pydantic ensures that this happens, but we can do it as a quick sanity check. 

If either of these two checks fail, we route to "Nullify Claim Form". Otherwise, we route to "Conditional Node: Does queue have documents?"

#### Nullify Claim Form
If the customer verification checks fail, we must delete all information from claim form fields in our state object. For audit purposes, we should keep the document and its OCR output in the database, but the state must be reset for the rest of the architecture to work as intended. The idea is that fields such as the one that "Conditional Node: Is claim form is state?" are reset, so that on the next document upload, we can route to the binary classification, as if nothing happened with the current faulty claim form. After nullifying this, we can END and wait for the next document upload to retrigger the graph.

#### Conditional Node: Does queue have documents?
This is a conditonal node that checks whether the length of the queue is greater than 0. If so, then we route to the "Multi-Classifier" with the parallelization optimizations. Otherwise, we can END and wait for the next document upload.

#### Multi-Class Classifier
Once we reach this node, the claim form and all its relevant information must be in our state. This is the first node in a two step sequence (OCR classification and structured output) that can be parallelized depending on how many inputs we are processing (number of documents in the queue + current document, if it is not the claim form). First, we classify the document into given classes. In most cases, users will upload each document type in separate files (one upload of the receipts, one upload of the boarding pass, one upload of the passport...), however, there are certain documents that are commonly grouped together, for instance, in the case of the trip cancellation claim, the proof of payment and the invoices may appear in one document. In order to account for this scenario, we anticipate this via multi-class classification. Previous iterations of the agent would assign a single class per document. However, it is smart to (if necessary) assign multiple classes per document. 

The reason for doing this step after the claim form is received is that we want to make sure that we only assign the classes relevant to the claim type. Some documents are used across all classes (claim form, passport, itinerary), however, others are exclusive to certain types of claims, such as PIR for baggage delay claims. Therefore, we conditionally assign classes depending on the claim type. This is easily implemented by defining a list of classes per claim type, and dynamically editing a `Pydantic` schema that forces the LLM response to be a subset of that list of classes. Ideally, we would also retrieve the line numbers pertaining to each class, though more research is needed to check if LLM can reliably do that.

#### Structured Output
The structured output node runs once for each identified class of each document. This information is stored in the state object, and potentially the database if needed for audit and checkpointing purposes. Structured output is performed by `gemini-2.5-flash` with inputs being the document OCR text and the appropriate `Pydantic` schema. These schemas are predetermined by us per document type and work together with the state object. This allows us to ensure we consistently gather all the information we need per document.   


#### Conditional Node: Is this document's checklist complete?

This conditional node checks whether we have at least some sort of information for each field in the uploaded document's checklist. If so, we route to "Conditional Node: Are all checklists complete?". Otherwise we route to "Nullify Document". 

#### Nullify Document

In the case where the uploaded document does not have all the information we are looking for, we must nullify said upload. ===We first ensure the document is stored in the database, then we reset the document checklist and accompanying fields.=== The idea is to essentially 'return' to a state before the document was uploaded, so that when the graph ENDs after this upload, a new upload of the same type can be processed without any information of the old upload tainting the new information.

This node and "Nullify Claim Form" could in theory be functionalized into one node. For the sake of simplifying the diagram, I have split it into two, but they essentially do the same thing.

#### Conditional Node: Are all checklists complete?

This conditional node checks whether all document checklists are complete, in other words, if we are ready to advance to Stage 2. If all checklists are complete, we can move onto stage 2. Otherwise, we can simply END and wait for the next document upload to trigger the graph. If all relevant documents are uploaded and we get to this stage, then we necessarily must move onto Stage 2.

### Stage 2

Stage 2 is what I envision as the 'thinking stage' of the graph, whereas Stage 1 is the 'user interaction stage' of this system. Once we arrive at Stage 2, we should have all documents uploaded and their supplementary checklists completed. In the provided graph drawing, there is only one orchestrator. This was done for the sake of keeping a simple diagram. However, in practice, we may want multiple types of orchestrators (one for each claim type) so that we can more adequately call tools and subagents with proper context. For example, it could be the case where a certain document analysis function must only happen for a given claim type. Instead of having messy conditional logic at the subagent level, we can orchestrate it beforehand. This makes our tool more modular as we limit the effects of claim types to certain nodes that can be edited if necessary. If we were to have conditional cases for claim types at each node, it would become a nightmare to remove or add a node. With this architecture we avoid that headache. 

Going back on track now, once we enter Stage 2, we should be at a stage where the agent could return some sort of recommendation to the claims handler, unless there is a critical problem (receipts are unreconcilable, passport is from a different person, etc). 

#### Claim Type Orchestrator

We begin Stage 2 with a router node that redirects flow depending on claim type. For every claim type we support, we should have one orchestrator responsible for said type. For example, in the Zurich case, we would like to support four claim types: travel delay, baggage delay, trip cancellation, and trip interruption. This means that under this architecture we would require four orchestrators. Each claim type orchestrator have all of the same edges: "Report Failure & Request Document(s)", "Reconciliation Tool", "Document Analysis Subagent", and "Final Processing Tool". However, the information that these orchestrators send through these edges may vary--all of these send the state object, however, they may also send additional inputs to nodes that influences their functionality. As briefly mentioned, we could add a boolean flag that forces the "Document Analysis Subagent" to use certain tools, among other interesting cases. 

#### Reconciliation Workflow

The reconciliation workflow is a set of nodes used to match up the expenses listed in the claim form with those in the uploaded receipts. We begin by trying to match each expense naively. If this process fails, we then try the intelligent reconciliation method using LLM. If either of these are successful, then the state is updated with the reconciled information and we can return to the Orchestrator in order to continue with Document Analysis. If both the naive matching and intelligent reconciliation fails, then we must send an error to the Orchestrator, which will then relay the failure to the user and request new documents to fix said error.

##### Naive Match Expenses

The Naive Match expenses is quite a simple node: we simply try to match each item in the claim form with some item in the receipts. We try this before the intelligent reconciliation because it is just a simple check and happens pretty much intantly and could mean one less LLM call. If this fails, we return an error to the reconciliation director node.

##### Intelligent Reconciliation

Upon failing "Naive Match Expenses", we arrive at the intelligent reconciliation node which uses a reasoning model to figure out the relationship between the claim form and the receipts. The idea is that the previous node can reconcile simple cases (few expenses, receipts exactly match claim form). This node uses a reasoning model (currently `gemini-2.5-pro`) to come up with a vector $\alpha$ which contains factors that line up with each receipt line item to match up with the claim form. 

We have seen that Gemini-2.5-pro, when given all relevant user files and the policy wording can correctly process a claim, giving correct payout and reasoning for its decisions. The LLM-based reconciler node leverages the power of reasoning models to link receipts and invoices to the claim form in difficult cases (e.g. "I only used 6/10 nights of my hotel stay for trip cancellation claims", or, ==="I split the dinner bill with my friend"===, etc). See the prompt below for a better understanding of what the expectations for this node are. Another important aspect is that the output of this node is an object containing a verifiable $\alpha$ vector using the following formula: 

$$\text{claimed expense}_i = \alpha_i \cdot \text{receipt expense}_i$$

$$\begin{bmatrix} 
\text{claimed expense}_1 \\ 
\text{claimed expense}_2 \\ 
\vdots \\ 
\text{claimed expense}_n 
\end{bmatrix} = 
\begin{bmatrix} 
\alpha_1 & 0 & \cdots & 0 \\ 
0 & \alpha_2 & \cdots & 0 \\ 
\vdots & \vdots & \ddots & \vdots \\ 
0 & 0 & \cdots & \alpha_n 
\end{bmatrix} 
\begin{bmatrix} 
\text{receipt expense}_1 \\ 
\text{receipt expense}_2 \\ 
\vdots \\ 
\text{receipt expense}_n 
\end{bmatrix}$$

```markdown
You are ClaimRecon AI, a reasoning engine that reconciles travel-insurance claims against provided documents.
You will receive a set of documents, including a claim form, proofs of payment (receipts, itineraries), proofs of loss (cancellations), and the insurance policy summary. Your task is to analyze these documents and produce a structured reconciliation report in JSON format.
Your Tasks & Rules
1. Extract Candidate Data
Claimed Expenses: From the claim form, parse every line item the claimant is seeking reimbursement for.
Proof of Payment: From receipts and the original trip itinerary, parse every line item that shows a cost paid by the claimant.
Proof of Loss: From cancellation emails or supplier letters, extract information confirming that a pre-paid service was unused and non-refundable.
2. Reconcile and Validate
Currency Validation: Verify that all monetary values for a given expense are in the single policy currency (USD). If any associated value (claim, receipt, or proof of loss) is in a different currency, the reconciliation for that item must be marked as FAILED.
Eligibility Check: Reconcile all claimed expenses, even those explicitly ineligible under the policy. Do not discard them. Instead, use the is_eligible flag and reason_for_ineligibility field in the output to mark them as not covered.
Ignore Policy Limits: Do not apply any policy limits (e.g., per-diem caps, maximum benefit amounts). The reconciliation should reflect the full claimed and proven
amounts.
3. Apply Reconciliation Logic
For every Claimed Expense, find the corresponding Proof of Payment line(s) and apply one of the following rules:
Rule A: Partial Use of Pre-Paid Service (e.g., unused hotel nights, cancelled tours)
The reconciliation MUST link back to the total amount on the original Proof of Payment (e.g., the "Hotels (10 nights): $2,500" line in the master itinerary). Calculate a coefficient α representing the unused, non-refundable fraction. The Proof of Loss is used to confirm the non-refundable status and find the value of the unused portion needed to calculate α.
Rule B: New / Additional Expense (e.g., last-minute flight home)
The reconciliation should be a direct, one-to-one match (α = 1.0) with its own Proof of Payment (e.g., the new flight's receipt).
4. Generate Structured Output
Return ONLY a single, valid JSON object. Provide no additional introductory text or commentary.
The root of the object must be a key named reconciliation_pairs, which contains a list of objects.
Each object in the list represents a single claimed expense and must follow the structure detailed in the example below.
Example JSON Output Structure
This example illustrates the required fields and logic for a single successfully reconciled expense.
JSON
{
    "reconciliation_pairs": [
        {
            "claimed_expense": {
                "description": "Unused Hotel San Marco (Venice)",
                "amount": 750,
                "currency": "USD"
            },
            "status": "SUCCESS",
            "reason_for_failure": null,
            "is_eligible": true,
            "reason_for_ineligibility": null,
            "matched_receipts": [
                {
                    "coefficient": 0.3,
                    "receipt_line": {
                        "description": "Bella Italia Itinerary - Hotels (10 nights)",
                        "amount": 2500,
                        "currency": "USD",
                        "source_document": "itinerary.txt"
                    },
                    "logic": {
                        "explanation": "Coefficient is the ratio of the non-refundable portion of the hotel cost ($750.00) to the total pre-paid hotel package cost ($2,500.00). This represents the 3 unused nights out of 10 total pre-paid nights.",
                        "source_files": [
                            "proof_of_unused_services.txt",
                            "itinerary.txt"
                        ]
                    }
                }
            ]
        }
    ]
}
```

#### Document Analysis Subagent

This subagent is only called after the orchestrator accepts the reconciliation outcome. This node is responsible for parallelizing execution of document analysis tools. These tools include but are not limited to passport fraud check, sensible purchases check (making sure no one claims designer clothing as essential, etc), receipt geolocation check (evaluates distance from customer's airport, useful for travel delay cases), timeline checks (establishes a timeline of events and makes sure receipts are not from before the trip or after the trip), flight existence verification (via FlightRadar24 API), etc. The idea is that each of these checks are completely modular and independent from each other so that the solution can adapt to the claim type and insurance plan needs more clearly. I will not go into detail on each of these tools because they are quite simple compared to the rest of the graph. These tools should be implemented in a way where inputs and outputs are clearly defined to ensure the modularity of the graph. 

Parallelization could occur with default `async/await` design patterns, but LangGraph gives us a simpler and more useful tool: `Send`. This function has the following signature: `Send("node_name", state)` where "node_name" is the name of the node that you want to parallelize, and `state` could be any dictionary. Using conditional edges, you can return a list of `Send`s which then execute in parallel automatically. From my understanding, even if the each parallelized node does not have the global state, it is not lost. There is some sort of automatic merging of `Send` outputs with the global merge. In cases where the nodes are writing to the same node, we may have to reconcile the edits manually. For instance if parallelized nodes A and B meet at node C, but both A and B write to a field called number, then we must use Reducers/operators to facillitate those edits. 

Using this `Send` function, we must collect the results using another node (this is not represented in the diagram). This node will use the Reducers and operators to merge all of new analyses with the global state. Maybe a clearer picture of the architecture is:

> subagent node (calls `Send` list) -> parallelized functions -> Collector node (merges changes with global state) -> Orchestrator

However, this really depends on the `State` object and at which stages we want to verify results. Some approaches could have another node after the Collector node which can rerun the subagent if any problem arised at the parallelization stage.

At the very least, the `State` object for this graph should have a subobject responsible for document analysis which is defined according to claim type. One of these fields should be `errors: str list` (or another more complex type where error messages can be relayed, along with their severity, etc) which can be used if there is any document analysis tool fails. 

#### Orchestrator Router

This is not a necessary node, but it is a nice way of simplifying the graph. Instead of having conditional edges routing by claim type after each workflow, subagent or tool finishes, it is best to have a single directed edge to an Orchestrator Router, which will then have those conditional edges routing back to the correct orchestrator. Both methods are possible, but using a 'graph-wide' orchestrator router reduces the edge number by quite a lot, especially in cases where we have many claim types. This also makes the system more modular, since additions or removals to a certain claim type only need changes at the Orchestrator level and not within any tool or workflow.

#### Final Processing Tool

After document analysis succeeds, the orchestrator routes the graph to the Final Processing Tool. There are three main parts here: "Verify Eligibility", "Apply Limits" and "Generate Recommendation".

##### Verify Eligibility

This node is left quite vague in purpose. The idea is that if any final checks after document analysis need to occur, they can be centralized in this node. 

##### Apply Limits

This node takes the reconciled expenses vector and applies limits to the final payout number. Also, if any non-reasonable expenses were detected at the document analysis step, we remove those from the payout here as well.

##### Generate Recommendation

This node essentially concludes the entire claims processing and returns a recommendation for a human claim-handler to read and accept or deny. The Generate Recommendation step is important, because since we are using AI, we cannot be certain that our output is correct for all cases. The tool therefore does not fully process a claim, it requires a human to take accountability for the outcome of the claim, but the AI does a lot of preliminary work and synthesizes said work for the human. 

The output of this tool is a recommendation, including a summary of the claim, confidence percentage based on the results of document analysis and reconciliation, a history of uploaded documents and any supplementary clarifying information from the insured.

#### Report Failure & Request Document(s)

In the case of a critical error in reconciliation or document analysis where it is clear that the uploaded documents do not match up with the claim (e.g. incorrect receipt upload, wrong passport, etc), we need to report that failure to the user by sending them a message about the situation: "The passport you uploaded does not match with the name of the person filling the claim form". Ideally, there could be a chat back and forth with the user that allows them to clarify that mistake: "I uploaded my wife's passport instead of mine, here is mine: {passport upload}". This information would be relayed to the claims handler once the entire claim is processed.

As we report the failure, we have to ensure that the upload is stored in the database, linked to the claim id. Once this has been verified, we can delete the document's information from the state. The state of the graph would be reset as if that document has never been uploaded. This means that after we END and the graph is retriggered by a document upload, we start back from Stage 1. Once we are back in Stage 1, the checklist condition will not be satisfied and we go through the document classification and structured output steps again until we get a new upload of the document we are expecting. 

## Conclusion

The described architecture is as streamlined as I could make it in the amount of time I had. I think that one of the strengths of this architecture is that it replicates the task that a human claims handler would have to do. Of course, there are some differences, namely the parallelization of document analyses and making some humanly implicit checks more explicit. Another strength is that I think this architecture pairs extremely well with LangGraph. I imagine it would also work in similar stateful agent frameworks.

As for weaknesses, I think that one of the largest weaknesses is that ideally, we should check for matching passport and claim form (i.e. there is no need to wait for document analysis to do it). This would save time and effort as we could prevent more claims from switching from Stage 1 to Stage 2 only for them to fail and have to return to Stage 1. Another weakness is that the diagram does not depict a lot of nice shortcuts and subtle optimizations that reduce the amount of work to do. A clear example is in the case where we have a critical Stage 2 error, lets say the passport verification failed, when the user uploads their new passport, it is not necessary to do the Reconciliation step again. We can move straight ahead to the passport check, and even skip the analyses that had good results. Another possible optimization is to make the "Nullify Claim Form", "Nullify Document", and "Relay Failure & Request Document" all be the same function albeit more sophisticated. Depending on the type of failure, it could return different and specific chat messages (maybe using LLM like `gemini-2.5-flash` or weaker) and nullify the state for only the document at hand.  











