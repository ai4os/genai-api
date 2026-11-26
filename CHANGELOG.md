# Changelog

## [1.1.0](https://github.com/ai4os/genai-api/compare/v1.0.1...v1.1.0) (2026-11-26)


### Features

* add credit usage endpoint ([a97830a](https://github.com/ai4os/genai-api/commit/a97830a40f012c12cbe8de5cedf65ab1ee4ba8bd))
* add delete all chats endpoint ([58aa088](https://github.com/ai4os/genai-api/commit/58aa088c81f5f2bd7389eb4987ca04f4e153102c))

## [1.0.1](https://github.com/ai4os/genai-api/compare/v1.0.0...v1.0.1) (2026-11-20)


### Bug Fixes

* add GENAI_KEY_NAME support (dev/prod) and implement key regeneration logic in LiteLLMClient ([fb617d2](https://github.com/ai4os/genai-api/commit/fb617d2d5cb4f82172be81b7e9806eac86ed9144))

## 1.0.0 (2026-11-17)


### Features

* add agents endpoint ([dc0416e](https://github.com/ai4os/genai-api/commit/dc0416eb5c82502be75ac2a6270f2b8f60c82c84))
* add base API with connection to a local PostgreSQL DB ([26e690f](https://github.com/ai4os/genai-api/commit/26e690f960ec1e62f8bcaa9d6774ed44dd44180e))
* add completions endpoints (streaming and non-streaming) ([09bc02c](https://github.com/ai4os/genai-api/commit/09bc02c2ef8090ec47acf8887289ce7b48615d82))
* add conversation history to agents ([f02ec02](https://github.com/ai4os/genai-api/commit/f02ec02be6fe68176c62a6f9d618805bae6c62c8))
* add images management and configure streaming for agents ([527a09f](https://github.com/ai4os/genai-api/commit/527a09f6c43f4bd5723a543b280f2856bb7ee509))
* add metadata to models info and cache results ([1220241](https://github.com/ai4os/genai-api/commit/1220241ff9e9276a1d58c8542fdc9bc50daa5142))
* add models endpoint and manage user LLM api key ([24e24dc](https://github.com/ai4os/genai-api/commit/24e24dce1229ff54f9a52a63cc18f5da44c955ee))
* add support for deleting chats ([a385579](https://github.com/ai4os/genai-api/commit/a385579459ada380fd05a91faa2b0e71682b522d))
* **agents:** send the complete conversation history excluding the last message ([78054b7](https://github.com/ai4os/genai-api/commit/78054b76183f5074e1140e720a980d2044bc6225))
* configure pre commit, add tests and update README ([3607c62](https://github.com/ai4os/genai-api/commit/3607c62c00ad5195237a3000a811b8eab76377d4))
* exclude default submodels from UI ([ecd9a8a](https://github.com/ai4os/genai-api/commit/ecd9a8a907833f5e188a5dd11559677331b9d71e))
* include more metadata in the agents info ([51edf6d](https://github.com/ai4os/genai-api/commit/51edf6d6879cfafff7416d705bb20ba82e5ab4da))
* integrate milvus mcp to llm calls ([2332193](https://github.com/ai4os/genai-api/commit/233219354d5c49287be622e1e480e26d8c1c1dd7))
* paginate collection files ([58f19d8](https://github.com/ai4os/genai-api/commit/58f19d8138a69bb9e0f8ad70ae4877242387f787))
* support for creating Milvus collections ([0f165de](https://github.com/ai4os/genai-api/commit/0f165de5982ebae7dfa8bf18c5518d5a2290bba5))
* support listing/deleting files  from a particular collection ([1d30a4d](https://github.com/ai4os/genai-api/commit/1d30a4d4f4dd70f9c52181b116e8727cee32b8f1))


### Bug Fixes

* circular import ([b00fb60](https://github.com/ai4os/genai-api/commit/b00fb60317a939facf1d1623ae2a6b44506b8002))
* incorrect model list issue ([5952aab](https://github.com/ai4os/genai-api/commit/5952aabf180f60577faf591f211de959a5189cd7))
* race condition creating new users ([c2b4069](https://github.com/ai4os/genai-api/commit/c2b4069f97e2b197488a37d0377da3cd5c387076))
* use heartbeat to prevent timeout issues ([ba897aa](https://github.com/ai4os/genai-api/commit/ba897aa8fd04c1b237e323a4881bf3abc30ce0c5))
