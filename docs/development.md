# Development and hosting

## Run tests and build distributions

```bash
python -m pip install -e '.[dev]'
LITELLM_LOCAL_MODEL_COST_MAP=True python -m pytest
python -m build
```

The normal suite makes no provider requests and needs no device permissions.
Live-provider tests are opt-in: `RUN_LIVE_TESTS=1 python -m pytest
tests/test_live_providers.py`. They require configured credentials and incur
provider charges. The model names are configured in that test file.

CI runs the offline suite on Python 3.11, 3.12, 3.13, and 3.14, and builds the
source distribution and wheel. It installs Linux media system libraries, but
does not open a camera, microphone, or keyboard listener.

## Generate documentation

The documentation build needs only Python and the documentation requirements;
the runtime package and its media/provider dependencies need not be installed.

```bash
python -m pip install -r requirements-docs.txt
python -m mkdocs serve
```

Open the local address printed by MkDocs. For a deployable build:

```bash
python -m mkdocs build --strict
```

The output is `site/`, which is ignored by Git. The API reference is generated
from source by [mkdocstrings](https://mkdocstrings.github.io/python/usage/), with
runtime inspection disabled. New public methods appear automatically on their
class reference page. Add new
public classes/modules to `docs/reference/` and `mkdocs.yml`.

## Publish on GitHub Pages

GitHub Pages supports public repositories on GitHub Free. If you keep a fork
private, Pages requires an eligible paid plan, and the resulting site is normally
public even when the source repository is private. See
[GitHub's availability and visibility rules](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site).

The repository contains `.github/workflows/docs.yml`. Pull requests build the
site without deployment. Pushes to `main` and manual runs on `main` build and
deploy it through GitHub's Pages artifact workflow.

1. Commit and push the source, docs, requirements, and workflows to GitHub.
2. In the repository's **Settings → Pages → Build and deployment**, select
   **GitHub Actions** as the source. This is a one-time repository setting.
3. Run **Documentation** from the Actions tab, or push another commit to `main`.
4. Open the deployment URL shown by the `github-pages` environment. For this
   repository, the site is
   [alistairmacvicar.github.io/llm-toolkit](https://alistairmacvicar.github.io/llm-toolkit/).

The workflow needs `pages: write` and `id-token: write` only for deployment.
It uses the repository's `GITHUB_TOKEN`; no personal access token is needed.
See GitHub's [custom Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

For a fork, update `site_url`, `repo_url`, and the URLs in `pyproject.toml`.
The generated site is static HTML; it does not host a running Python application.
