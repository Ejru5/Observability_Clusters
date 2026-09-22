# Cluster Report

**Total spans:** 62  |  **Macro clusters:** 7  |  **Noise (unclustered):** 2

---

## [6] Truncated Financial Analysis Responses  --  10 spans
> The model consistently generates incomplete outputs due to hitting token length limits across mortgage verification, underwriting, and compliance analysis prompts.

  - **[6.1]** Truncated Compliance Audit Reports  --  6 spans
    > The model consistently generates incomplete financial compliance audit reports due to hitting the output token length limit before completing the response.
  - **[6.0]** Truncated Underwriting Analysis Responses  --  4 spans
    > The model consistently generates incomplete financial analysis outputs due to hitting the maximum token limit before completing the response.

## [0] Truncated Real Estate Analysis Responses  --  9 spans
> The model consistently generates incomplete outputs due to hitting token length limits while producing real estate legal or investment analysis.

  - **[0.1]** Truncated Legal Due Diligence Reports  --  4 spans
    > The model consistently generates incomplete legal analysis responses for commercial real estate lease documentation due to hitting output length limits.
  - **[0.0]** Truncated Legal Due Diligence Analysis  --  2 spans
    > The model consistently truncates responses mid-analysis for commercial real estate legal due diligence prompts due to output length constraints.
  - **[0.-1]** [NOISE] [noise]  --  3 spans
    > Unclustered spans within this macro group.

## [3] High Latency Workflow Timeouts  --  9 spans
> Multiple workflow operations consistently exceed P95 latency thresholds without explicit errors, indicating systemic performance degradation or resource contention.

  - **[3.0]** High Latency Workflow Timeouts  --  4 spans
    > Multiple workflow operations consistently exceed P95 latency thresholds without explicit errors, indicating systemic performance degradation or resource contention.
  - **[3.1]** High Latency Workflow Processing Failures  --  3 spans
    > Multiple workflow operations consistently exceed latency thresholds or exhibit structural anomalies without explicit errors, indicating systemic performance degradation or input processing issues.
  - **[3.-1]** [NOISE] [noise]  --  2 spans
    > Unclustered spans within this macro group.

## [4] Truncated Luxury Travel Correspondence  --  9 spans
> The model consistently generates incomplete responses for high-touch customer communications, particularly luxury travel concierge and warranty correspondence, due to output length constraints.

  - **[4.1]** Truncated Luxury Travel Correspondence  --  6 spans
    > The model consistently generates incomplete responses due to hitting the output length limit while drafting formal travel concierge communications.
  - **[4.0]** Truncated Customer Correspondence Responses  --  3 spans
    > The model consistently generates incomplete responses due to hitting the maximum token limit before finishing the intended output.

## [5] Truncated Compliance Audit Responses  --  9 spans
> The model consistently generates incomplete outputs due to hitting the maximum token limit, cutting off critical compliance or audit-related responses mid-sentence.

  - **[5.1]** Truncated Compliance Audit Responses  --  4 spans
    > The model consistently generates incomplete compliance review outputs due to hitting the maximum token length limit before completing the statutory analysis.
  - **[5.0]** Truncated Audit Response Outputs  --  3 spans
    > The model consistently truncates structured audit responses due to hitting the maximum token limit before completing the required output.
  - **[5.-1]** [NOISE] [noise]  --  2 spans
    > Unclustered spans within this macro group.

## [1] Missing Wire Routing Data Parsing Failures  --  8 spans
> The spans consistently fail due to KeyError for 'wire_routing_number' or JSONDecodeError from malformed/empty API responses in mortgage settlement workflows.

  - **[1.0]** Missing Wire Routing Data Parsing Failures  --  4 spans
    > The spans consistently fail due to a KeyError for 'wire_routing_number', indicating missing or unparsed wire routing data in the mortgage settlement pipeline.
  - **[1.1]** Empty JSON Response Parsing Failures  --  4 spans
    > The agent consistently fails to parse empty or malformed JSON responses, triggering JSONDecodeError at the initial character position.

## [2] Tool Execution Connection Failures  --  6 spans
> All spans show failed tool executions with missing error details, indicating a systemic connectivity or authentication issue with external tool endpoints.

  - **[2.-1]** [NOISE] [noise]  --  6 spans
    > Unclustered spans within this macro group.

## [-1] [NOISE] [NOISE]  --  2 spans
> Unclustered spans with no clear pattern.

  *(no subclusters -- macro cluster too small)*
