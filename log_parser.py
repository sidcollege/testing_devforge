"""Multi-language log parser for automated issue creation.

Internal component for github_operation tool - NOT a separate tool.
Parses stack traces from Python, JavaScript, Java, and Go.
"""

import logging
import re
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum

from src.core.model_router import ModelRouter
from src.core.features import Feature, FeatureFlags

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    GO = "go"
    UNKNOWN = "unknown"


@dataclass
class StackTrace:
    """Parsed stack trace information"""
    language: Language
    error_type: str
    message: str
    file: Optional[str]
    line: Optional[int]
    function: Optional[str]
    stack_frames: List[dict]
    raw_trace: str


@dataclass
class ParsedIssue:
    """Parsed issue data ready for GitHub"""
    title: str
    body: str
    labels: List[str]
    stack_trace: StackTrace
    root_cause: Optional[str]


class LogParser:
    """Parse error logs into structured issue data"""
    
    def __init__(self):
        self.model_router = ModelRouter()
    
    async def parse(
        self,
        log: str,
        language: Optional[str] = None
    ) -> ParsedIssue:
        """Parse error log into issue data
        
        Args:
            log: Raw error log/stack trace
            language: Optional language hint
            
        Returns:
            ParsedIssue ready for GitHub issue creation
        """
        if not FeatureFlags.is_enabled(Feature.LOG_PARSING):
            # Fallback to basic parsing
            return self._fallback_parse(log)
        
        # Auto-detect language
        if not language:
            lang = self._detect_language(log)
        else:
            lang = Language(language.lower())
        
        # Parse stack trace
        if lang == Language.PYTHON:
            trace = self._parse_python_trace(log)
        elif lang == Language.JAVASCRIPT:
            trace = self._parse_js_trace(log)
        elif lang == Language.JAVA:
            trace = self._parse_java_trace(log)
        elif lang == Language.GO:
            trace = self._parse_go_trace(log)
        else:
            trace = self._parse_generic_trace(log)
        
        # Infer root cause using LLM
        root_cause = await self._infer_root_cause(trace)
        
        # Suggest labels
        labels = self._suggest_labels(trace, root_cause)
        
        # Format issue
        title = self._format_title(trace)
        body = self._format_body(trace, root_cause)
        
        return ParsedIssue(
            title=title,
            body=body,
            labels=labels,
            stack_trace=trace,
            root_cause=root_cause
        )
    
    def _detect_language(self, log: str) -> Language:
        """Auto-detect programming language from log
        
        Args:
            log: Raw log content
            
        Returns:
            Detected Language
        """
        # Python indicators
        if any(x in log for x in ["Traceback", "File \"", "line "]):
            return Language.PYTHON
        
        # JavaScript indicators
        if any(x in log for x in ["at ", ".js:", "Error:", "TypeError:"]):
            if "    at " in log:  # V8 stack trace format
                return Language.JAVASCRIPT
        
        # Java indicators
        if any(x in log for x in ["Exception in thread", ".java:", "at "]):
            if "Exception" in log and ".java:" in log:
                return Language.JAVA
        
        # Go indicators
        if "panic:" in log or "goroutine" in log:
            return Language.GO
        
        return Language.UNKNOWN
    
    def _parse_python_trace(self, log: str) -> StackTrace:
        """Parse Python traceback
        
        Example:
            Traceback (most recent call last):
              File "app.py", line 42, in main
                result = divide(10, 0)
            ZeroDivisionError: division by zero
        """
        lines = log.strip().split("\n")
        
        # Extract error type and message (last line)
        error_line = lines[-1]
        if ":" in error_line:
            error_type, message = error_line.split(":", 1)
            error_type = error_type.strip()
            message = message.strip()
        else:
            error_type = "Error"
            message = error_line
        
        # Parse stack frames
        stack_frames = []
        file_pattern = r'File "([^"]+)", line (\d+)(?:, in (.+))?'
        
        for i, line in enumerate(lines):
            match = re.search(file_pattern, line)
            if match:
                file, line_no, func = match.groups()
                stack_frames.append({
                    "file": file,
                    "line": int(line_no),
                    "function": func,
                    "code": lines[i + 1].strip() if i + 1 < len(lines) else None
                })
        
        # Get topmost frame
        top_frame = stack_frames[-1] if stack_frames else {}
        
        return StackTrace(
            language=Language.PYTHON,
            error_type=error_type,
            message=message,
            file=top_frame.get("file"),
            line=top_frame.get("line"),
            function=top_frame.get("function"),
            stack_frames=stack_frames,
            raw_trace=log
        )
    
    def _parse_js_trace(self, log: str) -> StackTrace:
        """Parse JavaScript stack trace
        
        Example:
            TypeError: Cannot read property 'x' of undefined
                at Object.<anonymous> (/app/index.js:10:5)
                at Module._compile (internal/modules/cjs/loader.js:1137:30)
        """
        lines = log.strip().split("\n")
        
        # First line is usually error type and message
        error_line = lines[0]
        if ":" in error_line:
            error_type, message = error_line.split(":", 1)
            error_type = error_type.strip()
            message = message.strip()
        else:
            error_type = "Error"
            message = error_line
        
        # Parse stack frames
        stack_frames = []
        frame_pattern = r'at\s+(?:(.+?)\s+)?\((.+?):(\d+):(\d+)\)'
        
        for line in lines[1:]:
            match = re.search(frame_pattern, line)
            if match:
                func, file, line_no, col = match.groups()
                stack_frames.append({
                    "function": func or "<anonymous>",
                    "file": file,
                    "line": int(line_no),
                    "column": int(col)
                })
        
        top_frame = stack_frames[0] if stack_frames else {}
        
        return StackTrace(
            language=Language.JAVASCRIPT,
            error_type=error_type,
            message=message,
            file=top_frame.get("file"),
            line=top_frame.get("line"),
            function=top_frame.get("function"),
            stack_frames=stack_frames,
            raw_trace=log
        )
    
    def _parse_java_trace(self, log: str) -> StackTrace:
        """Parse Java stack trace
        
        Example:
            Exception in thread "main" java.lang.NullPointerException
                at com.example.App.main(App.java:15)
        """
        lines = log.strip().split("\n")
        
        # First line is error
        error_line = lines[0]
        if "Exception" in error_line:
            parts = error_line.split()
            error_type = parts[-1].split(".")[-1] if "." in parts[-1] else parts[-1]
            message = " ".join(parts[:-1]) if len(parts) > 1 else error_type
        else:
            error_type = "Exception"
            message = error_line
        
        # Parse stack frames
        stack_frames = []
        frame_pattern = r'at\s+(.+?)\((.+?\.java):(\d+)\)'
        
        for line in lines[1:]:
            match = re.search(frame_pattern, line)
            if match:
                func, file, line_no = match.groups()
                stack_frames.append({
                    "function": func,
                    "file": file,
                    "line": int(line_no)
                })
        
        top_frame = stack_frames[0] if stack_frames else {}
        
        return StackTrace(
            language=Language.JAVA,
            error_type=error_type,
            message=message,
            file=top_frame.get("file"),
            line=top_frame.get("line"),
            function=top_frame.get("function"),
            stack_frames=stack_frames,
            raw_trace=log
        )
    
    def _parse_go_trace(self, log: str) -> StackTrace:
        """Parse Go panic trace
        
        Example:
            panic: runtime error: invalid memory address
            goroutine 1 [running]:
            main.main()
                /app/main.go:10 +0x35
        """
        lines = log.strip().split("\n")
        
        # Extract panic message
        panic_line = next((l for l in lines if l.startswith("panic:")), "")
        if panic_line:
            message = panic_line.replace("panic:", "").strip()
            error_type = "panic"
        else:
            error_type = "error"
            message = lines[0] if lines else ""
        
        # Parse stack frames
        stack_frames = []
        for i, line in enumerate(lines):
            if "/" in line and ".go:" in line:
                # Function is on previous line
                func = lines[i - 1].strip() if i > 0 else ""
                
                # Parse file:line
                match = re.search(r'(.+\.go):(\d+)', line)
                if match:
                    file, line_no = match.groups()
                    stack_frames.append({
                        "function": func,
                        "file": file,
                        "line": int(line_no)
                    })
        
        top_frame = stack_frames[0] if stack_frames else {}
        
        return StackTrace(
            language=Language.GO,
            error_type=error_type,
            message=message,
            file=top_frame.get("file"),
            line=top_frame.get("line"),
            function=top_frame.get("function"),
            stack_frames=stack_frames,
            raw_trace=log
        )
    
    def _parse_generic_trace(self, log: str) -> StackTrace:
        """Fallback generic parsing"""
        return StackTrace(
            language=Language.UNKNOWN,
            error_type="Error",
            message=log[:200],
            file=None,
            line=None,
            function=None,
            stack_frames=[],
            raw_trace=log
        )
    
    async def _infer_root_cause(self, trace: StackTrace) -> Optional[str]:
        """Infer root cause using LLM
        
        Args:
            trace: Parsed stack trace
            
        Returns:
            Root cause analysis
        """
        try:
            model = await self.model_router.select_model_by_task("github")
            
            prompt = f"""Analyze this {trace.language.value} error and provide root cause:

Error: {trace.error_type}: {trace.message}
File: {trace.file}:{trace.line} in {trace.function}

Provide a 1-2 sentence root cause analysis. Be specific and actionable.
"""
            
            response = await self.model_router.invoke_with_fallback(
                model=model,
                prompt=prompt,
                fallback_chain=["gpt-oss:120b-cloud"]
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Failed to infer root cause: {e}")
            return None
    
    def _suggest_labels(
        self,
        trace: StackTrace,
        root_cause: Optional[str]
    ) -> List[str]:
        """Suggest GitHub issue labels
        
        Args:
            trace: Parsed stack trace
            root_cause: Root cause analysis
            
        Returns:
            List of label names
        """
        labels = ["bug"]
        
        # Add language label
        if trace.language != Language.UNKNOWN:
            labels.append(trace.language.value)
        
        # Add severity label based on error type
        critical_errors = ["NullPointerException", "SegFault", "panic", "FATAL"]
        if any(err in trace.error_type for err in critical_errors):
            labels.append("P0-critical")
        else:
            labels.append("P1-high")
        
        # Add component label from file path
        if trace.file:
            if "auth" in trace.file.lower():
                labels.append("auth")
            elif "api" in trace.file.lower():
                labels.append("api")
            elif "db" in trace.file.lower() or "database" in trace.file.lower():
                labels.append("database")
        
        return labels
    
    def _format_title(self, trace: StackTrace) -> str:
        """Format issue title
        
        Args:
            trace: Parsed stack trace
            
        Returns:
            Issue title
        """
        # Format: [ErrorType] Brief message (file:line)
        title = f"[{trace.error_type}] {trace.message[:50]}"
        
        if trace.file and trace.line:
            filename = trace.file.split("/")[-1]
            title += f" ({filename}:{trace.line})"
        
        return title[:80]  # GitHub title limit
    
    def _format_body(
        self,
        trace: StackTrace,
        root_cause: Optional[str]
    ) -> str:
        """Format issue body
        
        Args:
            trace: Parsed stack trace
            root_cause: Root cause analysis
            
        Returns:
            Issue body in markdown
        """
        body = f"""## Error Details

**Type:** `{trace.error_type}`  
**Message:** {trace.message}  
**Language:** {trace.language.value}
"""
        
        if trace.file:
            body += f"**Location:** `{trace.file}:{trace.line}`"
            if trace.function:
                body += f" in `{trace.function}()`"
            body += "\n\n"
        
        if root_cause:
            body += f"""## Root Cause Analysis

{root_cause}

"""
        
        body += f"""## Stack Trace

```{trace.language.value}
{trace.raw_trace}
```
"""
        
        return body
    
    def _fallback_parse(self, log: str) -> ParsedIssue:
        """Fallback basic parsing when log parsing disabled"""
        return ParsedIssue(
            title=f"Error: {log[:50]}",
            body=f"```\n{log}\n```",
            labels=["bug"],
            stack_trace=self._parse_generic_trace(log),
            root_cause=None
        )
