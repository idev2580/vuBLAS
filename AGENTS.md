# vuBLAS
A simple BLAS library based on vucol.

## Conventions
- Before do actual development, always write and summarize the user prompt into prompts directory as a markdown file, named `dev-prompt-{number}.md`. Count the number from 0.
    - Summary might be changed during the conversation. Summary should contain 'what to implement' and 'how to implement'
- Except for `prompts` directory, if you want to change any file, you must ask the user to allow that change.
- Do not run any compile or execution in agent's environment. That's developer's work.
- Do not use any additional external libraries. Only use the libraries already in use. For example, vucol, gtests, vk bootstrap(dependency of vucol), etc.
