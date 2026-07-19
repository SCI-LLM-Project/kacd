from pathlib import Path

import pytest

import config
from util.markdown_parser import read_markdown_file, process_markdown_paper

# The actual papers read during the experiment (config.DIRECTORY), not
# hand-crafted text.
REAL_PAPER_PATHS = sorted(Path(config.DIRECTORY).glob("**/*.md"))

REALISTIC_BODY = "\n\n".join(
    f"## Section {i}\n\n" + ("This is a sentence of body text. " * 20)
    for i in range(1, 6)
)


@pytest.mark.skipif(
    not REAL_PAPER_PATHS,
    reason=f"no markdown files found under config.DIRECTORY={config.DIRECTORY!r}",
)
class TestProcessMarkdownPaperOnRealCorpus:
    @pytest.mark.parametrize("path", REAL_PAPER_PATHS, ids=lambda p: p.stem)
    def test_process_markdown_paper_runs_without_error_and_is_nonempty(self, path):
        result = process_markdown_paper(str(path))
        assert isinstance(result, str)
        assert len(result) > 0


class TestReadMarkdownFile:
    def test_reads_file_contents(self, tmp_path):
        f = tmp_path / "paper.md"
        f.write_text("# Hello\n\nWorld", encoding="utf-8")
        assert read_markdown_file(str(f)) == "# Hello\n\nWorld"


class TestProcessMarkdownPaper:
    def test_reads_and_strips_whitespace(self, tmp_path):
        content = f"  \n# Title\n\n{REALISTIC_BODY}\n\n## References\n[1] citation\n  \n"
        f = tmp_path / "paper.md"
        f.write_text(content, encoding="utf-8")

        result = process_markdown_paper(str(f))

        assert result == content.strip()  # content passes through untruncated
        assert result.startswith("# Title")
