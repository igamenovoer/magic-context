you are tasked to find info using tavily mcp, here is the preference of using tavily mcp in different modes

## Skim Mode

when you want to quickly find information about a topic, you can use the skim mode. 

```json
{
  "query": "(your query)",
  "max_results": 15, // can be larger than this
  "topic": "general",
  "include_raw_content": false
}
```

## Detail Mode

when you want to find in-depth information about a topic, you can use the detailed mode. 

```json
{
  "query": "(your query)",
  "search_depth": "advanced",
  "max_results": 5, // can be larger than this, but less than 10
  "topic": "general",
  "include_raw_content": true
}
```