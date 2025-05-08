# genai-workshop-READY2025


## Installation

1. docker compose build
2. docker compose up -d
3. Run this query in the SMP, http://localhost:52773/csp/sys/%25CSP.Portal.Home.zen?$NAMESPACE=IRISAPP&
```sql
Select top 5 
encounter_id, DESCRIPTION_MEDICATIONS 
FROM GenAI.encounters
order by VECTOR_DOT_PRODUCT(DESCRIPTION_MEDICATIONS_Vector, embedding('tylenol','sentence-transformers/all-MiniLM-L6-v2')) desc
```
