"""
Analyze episode diversity for SFT training assessment.
"""
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import re

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure stdout/stderr for UTF-8 on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

def analyze_diversity(input_file: Path):
    """Analyze the diversity of episodes in a JSONL file."""
    
    episodes = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    
    print(f"\n{'='*70}")
    print(f"  EPISODE DIVERSITY ANALYSIS")
    print(f"{'='*70}\n")
    
    print(f"Total episodes: {len(episodes)}\n")
    
    # 1. Length Analysis
    print(f"{'-'*70}")
    print("1. LENGTH DISTRIBUTION")
    print(f"{'-'*70}")
    
    user_lengths = []
    assistant_lengths = []
    total_lengths = []
    
    for ep in episodes:
        user_content = ""
        assistant_content = ""
        for msg in ep.get('messages', []):
            if msg['role'] == 'user':
                user_content += msg['content']
            elif msg['role'] == 'assistant':
                assistant_content += msg['content']
        
        user_lengths.append(len(user_content.split()))
        assistant_lengths.append(len(assistant_content.split()))
        total_lengths.append(len(user_content.split()) + len(assistant_content.split()))
    
    print(f"User message length (words):")
    print(f"  Min: {min(user_lengths)}, Max: {max(user_lengths)}, "
          f"Avg: {sum(user_lengths)/len(user_lengths):.1f}, "
          f"Median: {sorted(user_lengths)[len(user_lengths)//2]}")
    
    print(f"Assistant response length (words):")
    print(f"  Min: {min(assistant_lengths)}, Max: {max(assistant_lengths)}, "
          f"Avg: {sum(assistant_lengths)/len(assistant_lengths):.1f}, "
          f"Median: {sorted(assistant_lengths)[len(assistant_lengths)//2]}")
    
    print(f"Total episode length (words):")
    print(f"  Min: {min(total_lengths)}, Max: {max(total_lengths)}, "
          f"Avg: {sum(total_lengths)/len(total_lengths):.1f}, "
          f"Median: {sorted(total_lengths)[len(total_lengths)//2]}")
    
    # 2. Instruction Pattern Analysis
    print(f"\n{'-'*70}")
    print("2. INSTRUCTION PATTERN ANALYSIS")
    print(f"{'-'*70}")
    
    question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does']
    imperative_words = ['explain', 'describe', 'list', 'create', 'write', 'generate', 'provide', 'give', 'show', 'tell', 'help', 'make', 'build']
    
    questions = 0
    imperatives = 0
    other = 0
    
    for ep in episodes:
        user_msg = ""
        for msg in ep.get('messages', []):
            if msg['role'] == 'user':
                user_msg = msg['content'].lower()
                break
        
        if '?' in user_msg:
            questions += 1
        elif any(user_msg.strip().startswith(word) for word in imperative_words):
            imperatives += 1
        else:
            # Check if starts with question word
            first_word = user_msg.strip().split()[0] if user_msg.strip() else ""
            if first_word in question_words:
                questions += 1
            else:
                other += 1
    
    print(f"Questions (with '?' or starts with question word): {questions} ({questions/len(episodes)*100:.1f}%)")
    print(f"Imperatives (explain, describe, create, etc.):     {imperatives} ({imperatives/len(episodes)*100:.1f}%)")
    print(f"Other (statements, contextual, etc.):              {other} ({other/len(episodes)*100:.1f}%)")
    
    # 3. Task Type Detection
    print(f"\n{'-'*70}")
    print("3. TASK TYPE DISTRIBUTION")
    print(f"{'-'*70}")
    
    task_keywords = {
        'explanation': ['explain', 'clarify', 'what is', 'what are', 'why', 'how does', 'understanding'],
        'code_generation': ['code', 'function', 'implement', 'write a', 'create a script', 'program'],
        'analysis': ['analyze', 'compare', 'evaluate', 'assess', 'review'],
        'creative': ['story', 'poem', 'creative', 'imagine', 'write about'],
        'formatting': ['format', 'list', 'bullet', 'checklist', 'table', 'structure'],
        'instruction': ['how to', 'steps', 'guide', 'tutorial', 'teach'],
        'summary': ['summarize', 'tldr', 'brief', 'overview', 'key points'],
        'advice': ['advice', 'suggest', 'recommend', 'should i', 'opinion'],
        'factual': ['fact', 'when', 'where', 'who', 'date', 'history'],
        'math_logic': ['calculate', 'solve', 'equation', 'logic', 'proof', 'algorithm'],
    }
    
    task_counts = defaultdict(int)
    
    for ep in episodes:
        user_content = ""
        for msg in ep.get('messages', []):
            if msg['role'] == 'user':
                user_content = msg['content'].lower()
        
        # Count tasks (one episode can have multiple task types)
        for task_type, keywords in task_keywords.items():
            if any(keyword in user_content for keyword in keywords):
                task_counts[task_type] += 1
    
    # Sort by count
    sorted_tasks = sorted(task_counts.items(), key=lambda x: x[1], reverse=True)
    
    for task, count in sorted_tasks:
        print(f"  {task:20s}: {count:3d} ({count/len(episodes)*100:.1f}%)")
    
    # 4. Vocabulary Richness
    print(f"\n{'-'*70}")
    print("4. VOCABULARY RICHNESS")
    print(f"{'-'*70}")
    
    all_user_words = []
    all_assistant_words = []
    
    for ep in episodes:
        for msg in ep.get('messages', []):
            words = re.findall(r'\b[a-z]+\b', msg['content'].lower())
            if msg['role'] == 'user':
                all_user_words.extend(words)
            elif msg['role'] == 'assistant':
                all_assistant_words.extend(words)
    
    unique_user_vocab = len(set(all_user_words))
    unique_assistant_vocab = len(set(all_assistant_words))
    
    print(f"User vocabulary:")
    print(f"  Total words: {len(all_user_words)}")
    print(f"  Unique words: {unique_user_vocab}")
    print(f"  Type-Token Ratio (TTR): {unique_user_vocab/len(all_user_words):.3f}")
    
    print(f"\nAssistant vocabulary:")
    print(f"  Total words: {len(all_assistant_words)}")
    print(f"  Unique words: {unique_assistant_vocab}")
    print(f"  Type-Token Ratio (TTR): {unique_assistant_vocab/len(all_assistant_words):.3f}")
    
    # 5. Topic Clustering (simple keyword-based)
    print(f"\n{'-'*70}")
    print("5. TOPIC DISTRIBUTION (Top Keywords)")
    print(f"{'-'*70}")
    
    # Common words to exclude
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'been',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                 'could', 'may', 'might', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 
                 'they', 'them', 'their', 'this', 'that', 'these', 'those', 'my', 'your',
                 'me', 'like', 'about', 'just', 'so', 'what', 'how', 'why', 'when', 'where'}
    
    user_word_freq = Counter([w for w in all_user_words if w not in stopwords and len(w) > 3])
    
    print("Top 30 user message keywords:")
    for word, count in user_word_freq.most_common(30):
        print(f"  {word:20s}: {count:3d}")
    
    # 6. Conversational Complexity
    print(f"\n{'-'*70}")
    print("6. CONVERSATIONAL COMPLEXITY")
    print(f"{'-'*70}")
    
    multi_turn = sum(1 for ep in episodes if len([m for m in ep.get('messages', []) if m['role'] in ['user', 'assistant']]) > 2)
    single_turn = len(episodes) - multi_turn
    
    print(f"Single-turn (1 user, 1 assistant):  {single_turn} ({single_turn/len(episodes)*100:.1f}%)")
    print(f"Multi-turn (multiple exchanges):    {multi_turn} ({multi_turn/len(episodes)*100:.1f}%)")
    
    # System prompt variation
    system_prompts = set()
    for ep in episodes:
        sys_prompt = ep.get('system', '')
        if sys_prompt:
            system_prompts.add(sys_prompt[:100])  # First 100 chars for uniqueness check
    
    print(f"\nUnique system prompts: {len(system_prompts)}")
    
    # 7. Assessment
    print(f"\n{'='*70}")
    print("  DIVERSITY ASSESSMENT FOR 750M MODEL (Phase 3a)")
    print(f"{'='*70}\n")
    
    score = 0
    max_score = 0
    feedback = []
    
    # Episode count (target: 200-500 for initial training)
    max_score += 20
    if len(episodes) >= 200:
        score += 20
        feedback.append(f"[+] Episode count ({len(episodes)}) is sufficient for initial SFT training")
    elif len(episodes) >= 150:
        score += 15
        feedback.append(f"[~] Episode count is borderline; 200+ would be more robust")
    else:
        score += 10
        feedback.append(f"[-] Episode count is low; consider augmenting to 300-500")
    
    # Task diversity (target: 5+ task types, each >10%)
    max_score += 20
    diverse_tasks = sum(1 for _, count in task_counts.items() if count/len(episodes) >= 0.10)
    if diverse_tasks >= 5:
        score += 20
        feedback.append(f"[+] Task diversity is excellent ({diverse_tasks} major task types)")
    elif diverse_tasks >= 3:
        score += 15
        feedback.append(f"[~] Task diversity is moderate ({diverse_tasks} major task types)")
    else:
        score += 10
        feedback.append(f"[-] Task diversity is low ({diverse_tasks} major task types)")
    
    # Vocabulary richness (TTR for user: target >0.15)
    max_score += 15
    user_ttr = unique_user_vocab/len(all_user_words) if all_user_words else 0
    if user_ttr >= 0.15:
        score += 15
        feedback.append(f"[+] Vocabulary is rich (TTR: {user_ttr:.3f})")
    elif user_ttr >= 0.10:
        score += 10
        feedback.append(f"[~] Vocabulary is moderate (TTR: {user_ttr:.3f})")
    else:
        score += 5
        feedback.append(f"[-] Vocabulary is limited (TTR: {user_ttr:.3f})")
    
    # Length distribution (target: avg 100-300 words/episode)
    max_score += 15
    avg_length = sum(total_lengths)/len(total_lengths)
    if 100 <= avg_length <= 400:
        score += 15
        feedback.append(f"[+] Episode length is well-balanced (avg: {avg_length:.0f} words)")
    elif 50 <= avg_length <= 500:
        score += 10
        feedback.append(f"[~] Episode length is acceptable (avg: {avg_length:.0f} words)")
    else:
        score += 5
        feedback.append(f"[-] Episode length may need adjustment (avg: {avg_length:.0f} words)")
    
    # Instruction pattern balance (target: diverse question/imperative mix)
    max_score += 15
    if 30 <= questions/len(episodes)*100 <= 70 and 20 <= imperatives/len(episodes)*100 <= 60:
        score += 15
        feedback.append("[+] Instruction patterns are well-balanced")
    else:
        score += 10
        feedback.append("[~] Instruction patterns could be more balanced")
    
    # Multi-turn conversations (target: 10-30% multi-turn)
    max_score += 15
    multi_turn_pct = multi_turn/len(episodes)*100
    if 10 <= multi_turn_pct <= 40:
        score += 15
        feedback.append(f"[+] Multi-turn ratio is appropriate ({multi_turn_pct:.1f}%)")
    elif multi_turn_pct < 10:
        score += 10
        feedback.append(f"[~] Few multi-turn conversations ({multi_turn_pct:.1f}%); single-turn focused")
    else:
        score += 10
        feedback.append(f"[~] High multi-turn ratio ({multi_turn_pct:.1f}%); may increase complexity")
    
    final_score = (score / max_score) * 100
    
    print(f"Overall Diversity Score: {final_score:.1f}/100\n")
    
    for item in feedback:
        print(f"  {item}")
    
    print(f"\n{'-'*70}")
    print("RECOMMENDATION")
    print(f"{'-'*70}\n")
    
    if final_score >= 85:
        print("[EXCELLENT] Dataset is highly diverse and ready for Phase 3a training.")
        print("   Your 750M model should develop strong conversational capabilities.")
        print("   Consider augmenting to 400-600 episodes for even more robustness.")
    elif final_score >= 70:
        print("[GOOD] Dataset has sufficient diversity for initial SFT training.")
        print("   Your 750M model should learn basic conversational patterns well.")
        print("   Augmenting to 500-800 episodes would improve generalization.")
    elif final_score >= 55:
        print("[MODERATE] Dataset is usable but could benefit from improvements.")
        print("   Consider augmenting to 400+ episodes and balancing task types.")
        print("   Training will work but may need follow-up fine-tuning.")
    else:
        print("[LIMITED] Dataset needs significant expansion or rebalancing.")
        print("   Strongly recommend augmenting to 500+ episodes before training.")
        print("   Consider adding more diverse task types and topics.")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    from core.banner import print_banner
    print_banner("MyPT Episode Diversity", "Dataset Quality Analysis Tool")
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_episode_diversity.py <input_jsonl_file>")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"Error: File {input_file} not found!")
        sys.exit(1)
    
    analyze_diversity(input_file)

