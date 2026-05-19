"""AI-powered commit message generation from diffs.

Internal component for github_operation tool - NOT a separate tool.
Generates Conventional Commits format messages using LLM.
"""

import logging
import re
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum

from src.core.model_router import ModelRouter
from src.core.features import Feature, FeatureFlags

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Conventional Commits change types"""
    FEAT = "feat"  # New feature
    FIX = "fix"  # Bug fix
    DOCS = "docs"  # Documentation only
    STYLE = "style"  # Code style (formatting, etc)
    REFACTOR = "refactor"  # Code refactoring
    PERF = "perf"  # Performance improvement
    TEST = "test"  # Adding/updating tests
    CHORE = "chore"  # Build/tooling changes
    CI = "ci"  # CI configuration
    BUILD = "build"  # Build system changes


@dataclass
class DiffAnalysis:
    """Analysis of git diff"""
    files: List[str]
    additions: int
    deletions: int
    summary: str
    file_types: set


@dataclass
class CommitMessage:
    """Generated commit message with metadata"""
    text: str  # Full commit message
    type: ChangeType
    scope: Optional[str]
    description: str
    body: Optional[str]
    confidence: float  # 0.0 to 1.0
    
    def to_conventional_format(self) -> str:
        """Format as Conventional Commits
        
        Returns:
            Formatted commit message
        """
        if self.scope:
            header = f"{self.type.value}({self.scope}): {self.description}"
        else:
            header = f"{self.type.value}: {self.description}"
        
        if self.body:
            return f"{header}\n\n{self.body}"
        
        return header


class CommitGenerator:
    """AI-powered commit message generator"""
    
    def __init__(self):
        self.model_router = ModelRouter()
    
    async def generate(
        self,
        repo: str,
        diff: str,
        max_length: int = 72,
        tenant_id: Optional[str] = None,
        integration_name: Optional[str] = None,
        user_id: Optional[str] = None  # NEW: Phase 4 analytics support
    ) -> CommitMessage:
        """Generate commit message from diff
        
        Args:
            repo: Repository name
            diff: Git diff content
            max_length: Maximum description length
            
        Returns:
            Generated CommitMessage
        """
        if not FeatureFlags.is_enabled(Feature.COMMIT_GENERATION):
            # Fallback to simple generic message
            return CommitMessage(
                text="chore: update files",
                type=ChangeType.CHORE,
                scope=None,
                description="update files",
                body=None,
                confidence=0.5
            )
        
        # Analyze diff
        analysis = self._analyze_diff(diff)
        
        # Infer change type
        change_type = self._infer_change_type(analysis)
        
        # Generate message using LLM
        model = await self.model_router.select_model_by_task("github")
        
        prompt = self._build_prompt(analysis, change_type, max_length)
        
        try:
            usage_result = await self.model_router.invoke_with_usage(
                model_name=model,
                prompt=prompt,
                fallback_models=["gpt-oss:120b-cloud"],
                tenant_id=tenant_id,
                integration_name=integration_name,
                task_type="github_commit_gen",
                user_id=user_id  # NEW: Pass user_id to ModelRouter
            )
            response = usage_result.content
            
            # Parse response
            commit_msg = self._parse_llm_response(response, change_type, analysis)
            
            logger.info(
                f"Generated commit message: {commit_msg.text}",
                extra={"confidence": commit_msg.confidence}
            )
            
            return commit_msg
            
        except Exception as e:
            logger.error(f"Failed to generate commit message: {e}")
            
            # Fallback to rule-based generation
            return self._fallback_generation(analysis, change_type)
    
    def _analyze_diff(self, diff: str) -> DiffAnalysis:
        """Analyze git diff content
        
        Args:
            diff: Raw diff string
            
        Returns:
            DiffAnalysis
        """
        files = []
        additions = 0
        deletions = 0
        file_types = set()
        
        for line in diff.split("\n"):
            # Extract filenames
            if line.startswith("+++") or line.startswith("---"):
                filename = line.split("/", 1)[-1] if "/" in line else line[4:]
                if filename and filename != "/dev/null":
                    files.append(filename)
                    
                    # Extract file type
                    if "." in filename:
                        ext = filename.rsplit(".", 1)[-1]
                        file_types.add(ext)
            
            # Count additions/deletions
            elif line.startswith("+") and not line.startswith("+++"):
                additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1
        
        # Generate summary
        summary = f"{len(files)} file(s) changed: +{additions} -{deletions}"
        
        return DiffAnalysis(
            files=files,
            additions=additions,
            deletions=deletions,
            summary=summary,
            file_types=file_types
        )
    
    def _infer_change_type(self, analysis: DiffAnalysis) -> ChangeType:
        """infer change type from diff analysis
        
        Args:
            analysis: DiffAnalysis
            
        Returns:
            ChangeType
        """
        files_str = " ".join(analysis.files).lower()
        
        # Check for specific patterns
        if any(x in files_str for x in ["test", "spec", "__test__"]):
            return ChangeType.TEST
        
        if any(x in files_str for x in ["readme", "doc", ".md"]):
            return ChangeType.DOCS
        
        if any(x in files_str for x in [".yml", ".yaml", ".github", "ci"]):
            return ChangeType.CI
        
        if any(x in files_str for x in ["package.json", "requirements.txt", "setup.py"]):
            return ChangeType.BUILD
        
        # Check if mostly deletions (might be refactor or cleanup)
        if analysis.deletions > analysis.additions * 2:
            return ChangeType.REFACTOR
        
        # Default to feat for new code, fix for modifications
        if analysis.additions > analysis.deletions:
            return ChangeType.FEAT
        else:
            return ChangeType.FIX
    
    def _build_prompt(
        self,
        analysis: DiffAnalysis,
        change_type: ChangeType,
        max_length: int
    ) -> str:
        """Build LLM prompt for commit message generation
        
        Args:
            analysis: DiffAnalysis
            change_type: ChangeType
            max_length: int
            
        Returns:
            Prompt string
        """
        return f"""Generate a Conventional Commits format commit message.

Files changed: {', '.join(analysis.files[:5])}
Change summary: {analysis.summary}
Inferred type: {change_type.value}

Requirements:
- Format: <type>(<scope>): <description>
- Description must be ≤{max_length} characters
- Use lowercase for description
- Be specific and actionable
- Infer meaningful scope from files

Respond with ONLY the commit message, no explanation.
"""
    
    def _parse_llm_response(
        self,
        response: str,
        change_type: ChangeType,
        analysis: DiffAnalysis
    ) -> CommitMessage:
        """Parse LLM response into CommitMessage
        
        Args:
            response: LLM response text
            change_type: Inferred change type
            analysis: DiffAnalysis
            
        Returns:
            CommitMessage
        """
        # Clean response
        text = response.strip()
        
        # Parse conventional commit format
        # Pattern: type(scope): description
        pattern = r"^(feat|fix|docs|style|refactor|perf|test|chore|ci|build)(?:\(([^)]+)\))?: (.+)$"
        match = re.match(pattern, text, re.IGNORECASE)
        
        if match:
            type_str, scope, description = match.groups()
            
            # Calculate confidence based on match quality
            confidence = 0.95 if scope else 0.90
            
            return CommitMessage(
                text=text,
                type=ChangeType(type_str.lower()),
                scope=scope,
                description=description.strip(),
                body=None,
                confidence=confidence
            )
        
        # Fallback if response doesn't match format
        return CommitMessage(
            text=text,
            type=change_type,
            scope=None,
            description=text[:72],
            body=None,
            confidence=0.70
        )
    
    def _fallback_generation(
        self,
        analysis: DiffAnalysis,
        change_type: ChangeType
    ) -> CommitMessage:
        """Fallback rule-based commit message generation
        
        Args:
            analysis: DiffAnalysis
            change_type: ChangeType
            
        Returns:
            CommitMessage
        """
        # Extract scope from first file
        scope = None
        if analysis.files:
            first_file = analysis.files[0]
            if "/" in first_file:
                scope = first_file.split("/")[0]
        
        # Generate description
        if len(analysis.files) == 1:
            description = f"update {analysis.files[0]}"
        else:
            description = f"update {len(analysis.files)} files"
        
        text = f"{change_type.value}"
        if scope:
            text += f"({scope})"
        text += f": {description}"
        
        return CommitMessage(
            text=text,
            type=change_type,
            scope=scope,
            description=description,
            body=None,
            confidence=0.60
        )
