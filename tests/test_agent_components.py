#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test agentic RAG components without a trained model.

Validates:
1. WorkspaceEngine
2. WorkspaceTools
3. Toolcall parsing
4. AgentController (with mock model)

Usage:
    python tests/test_agent_components.py
"""

import os
import sys
import json
import tempfile
import shutil

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_workspace(base_dir: str):
    """Create a test workspace with sample documents."""
    docs_dir = os.path.join(base_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    # Create test documents
    with open(os.path.join(docs_dir, "python.md"), "w") as f:
        f.write("# Python Guide\n\nPython is a programming language.\n")
    
    with open(os.path.join(docs_dir, "ml.txt"), "w") as f:
        f.write("Machine Learning\n\nML is a subset of AI.\n")
    
    print(f"Created test workspace at {base_dir}")
    return base_dir


def test_workspace_engine(workspace_dir: str):
    """Test WorkspaceEngine."""
    print("\n" + "=" * 50)
    print("TEST 1: WorkspaceEngine")
    print("=" * 50)
    
    from core.workspace import WorkspaceEngine
    
    # Create engine without index (just document management)
    engine = WorkspaceEngine(workspace_dir, os.path.join(workspace_dir, "index"))
    
    print(f"[OK] Engine created")
    print(f"  Documents: {engine.num_docs}")
    print(f"  Has index: {engine.has_index}")
    
    # List docs
    docs = engine.list_docs()
    print(f"\n[OK] Listed {len(docs)} documents:")
    for doc in docs:
        print(f"  - {doc.title} ({doc.doc_id[:8]}...)")
    
    # Get doc text
    if docs:
        text = engine.get_doc_text(docs[0].doc_id)
        print(f"\n[OK] Got document text ({len(text)} chars)")
    
    return engine


def test_workspace_tools(engine):
    """Test WorkspaceTools."""
    print("\n" + "=" * 50)
    print("TEST 2: WorkspaceTools")
    print("=" * 50)
    
    from core.workspace import WorkspaceTools
    
    tools = WorkspaceTools(engine)
    
    print(f"[OK] Tools created")
    print(f"  Available: {tools.list_tools()}")
    
    # Test list_docs
    result = tools.execute("workspace.list_docs", {})
    print(f"\n[OK] workspace.list_docs:")
    print(f"  Total: {result.get('total', 0)}")
    
    # Test get_doc
    docs = result.get("documents", [])
    if docs:
        doc_id = docs[0]["doc_id"]
        result = tools.execute("workspace.get_doc", {"doc_id": doc_id})
        print(f"\n[OK] workspace.get_doc:")
        print(f"  Title: {result.get('title')}")
        print(f"  Length: {result.get('length')} chars")
    
    # Test summarize
    result = tools.execute("workspace.summarize", {"text": "Python is a great programming language for beginners and experts alike."})
    print(f"\n[OK] workspace.summarize:")
    print(f"  Summary: {result.get('summary', '')[:100]}...")
    
    # Test unknown tool
    result = tools.execute("unknown.tool", {})
    print(f"\n[OK] Unknown tool handled:")
    print(f"  Error: {result.get('error', 'none')[:50]}")
    
    return tools


def test_parsing():
    """Test toolcall parsing."""
    print("\n" + "=" * 50)
    print("TEST 3: Toolcall Parsing")
    print("=" * 50)
    
    from core.agent.parsing import (
        find_toolcall, render_toolcall, render_toolresult,
        has_toolcall, TOOLCALL_OPEN, TOOLCALL_CLOSE
    )
    
    # Test render_toolcall
    block = render_toolcall("workspace.search", {"query": "python", "top_k": 5})
    print(f"[OK] render_toolcall:")
    print(f"  {block}")
    
    assert TOOLCALL_OPEN in block
    assert TOOLCALL_CLOSE in block
    assert "workspace.search" in block
    
    # Test has_toolcall
    assert has_toolcall(block) == True
    assert has_toolcall("no toolcall here") == False
    print(f"[OK] has_toolcall works")
    
    # Test find_toolcall
    text = f"I'll search for that. {block} Let me check."
    tc = find_toolcall(text)
    
    assert tc is not None
    assert tc.name == "workspace.search"
    assert tc.arguments["query"] == "python"
    assert tc.arguments["top_k"] == 5
    
    print(f"[OK] find_toolcall:")
    print(f"  Name: {tc.name}")
    print(f"  Args: {tc.arguments}")
    
    # Test render_toolresult
    result_block = render_toolresult({"documents": [{"text": "..."}], "total": 1})
    print(f"\n[OK] render_toolresult:")
    print(f"  {result_block[:80]}...")


def test_agent_controller(tools):
    """Test AgentController with mock model."""
    print("\n" + "=" * 50)
    print("TEST 4: AgentController (Mock Model)")
    print("=" * 50)
    
    from core.agent import AgentController
    from core.agent.parsing import render_toolcall
    
    # Mock model that returns a toolcall then an answer
    class MockModel:
        def __init__(self):
            self.call_count = 0
        
        def generate(self, prompt, max_new_tokens=100):
            self.call_count += 1
            
            if self.call_count == 1:
                # First call: return a toolcall
                tc = render_toolcall("workspace.list_docs", {})
                return prompt + tc
            else:
                # Second call: return final answer
                return prompt + "Based on the documents, you have 2 files in your workspace."
    
    mock_model = MockModel()
    controller = AgentController(mock_model, tools)
    
    print(f"[OK] Controller created")
    
    # Run agent
    result = controller.run([
        {"role": "user", "content": "What documents do we have?"}
    ], verbose=False)
    
    print(f"\n[OK] Agent run completed:")
    print(f"  Steps: {result.get('steps', 0)}")
    print(f"  Tool calls: {len(result.get('tool_calls', []))}")
    print(f"  Answer: {result.get('content', '')[:100]}...")
    
    # Check tool was called
    tool_calls = result.get("tool_calls", [])
    if tool_calls:
        print(f"\n[OK] Tool calls made:")
        for tc in tool_calls:
            print(f"  - {tc['name']}")


def test_serialization():
    """Test conversation serialization for SFT."""
    print("\n" + "=" * 50)
    print("TEST 5: Conversation Serialization")
    print("=" * 50)
    
    from core.special_tokens import SPECIAL_TOKEN_STRINGS
    
    # Check all required tokens exist
    required = [
        "myPT_system_open", "myPT_system_close",
        "myPT_user_open", "myPT_user_close",
        "myPT_assistant_open", "myPT_assistant_close",
        "myPT_toolcall_open", "myPT_toolcall_close",
        "myPT_toolresult_open", "myPT_toolresult_close",
    ]
    
    for token in required:
        assert token in SPECIAL_TOKEN_STRINGS, f"Missing token: {token}"
    
    print(f"[OK] All {len(required)} required tokens present")
    
    # Show token values
    print(f"\nToken values:")
    for token in required[:4]:  # Just show a few
        print(f"  {token}: {SPECIAL_TOKEN_STRINGS[token]}")


def main():
    print("=" * 60)
    print("Agentic RAG Component Test Suite")
    print("=" * 60)
    
    # Create temp workspace
    temp_dir = tempfile.mkdtemp(prefix="agent_test_")
    
    try:
        # Setup
        workspace_dir = create_test_workspace(temp_dir)
        
        # Run tests
        engine = test_workspace_engine(workspace_dir)
        tools = test_workspace_tools(engine)
        test_parsing()
        test_agent_controller(tools)
        test_serialization()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("=" * 60)
        
        print("\nNext steps:")
        print("  1. Train a model: python train.py --model_name base ...")
        print("  2. Create toolcall training data (JSONL)")
        print("  3. Train with: python train.py --dataset_dir data/tool_sft --config_file configs/sft2/toolchat.json")
        print("  4. Test agent: python scripts/workspace_chat.py --model_name my_agent")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()





