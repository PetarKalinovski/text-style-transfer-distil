def create_multistep_prompt(source_text, target_style):
    """
    Creates a multi-step reasoning prompt for text style transfer
    """

    prompt = f"""You are an expert at text style transfer. Transform text using systematic 4-step reasoning: Analysis, Planning, Execution, and Verification.

## Output Format
1. **ANALYSIS:** Identify all style-specific elements in the source text
2. **PLANNING:** Describe specific changes needed for the target style  
3. **EXECUTION:** Apply the planned changes step by step
4. **VERIFICATION:** Check that the result meets target style criteria
5. **[TRANSFERRED]:** Final transformed text

## Example: Informal → Formal

**Source Text:** "hey can u help me out with this thing?"
**Target Style:** formal

**ANALYSIS:** Identify informal elements:
- "hey" = casual greeting (informal)
- "u" = text speak abbreviation (informal)
- "help me out" = casual phrasing (informal)
- "this thing" = vague reference (informal)
- Missing proper punctuation (informal)

**PLANNING:** Required transformations:
- Replace "hey" → "Excuse me" (formal greeting)
- Replace "u" → "you" (proper spelling)
- Replace "help me out" → "assist me" (formal phrasing)
- Replace "this thing" → "this matter" (formal clarity)
- Add proper punctuation

**EXECUTION:** Applying changes:
- "hey" → "Excuse me," 
- "u" → "you" 
- "help me out" → "assist me" 
- "this thing" → "this matter" 
- Add punctuation 

**VERIFICATION:** Check result meets formal criteria:
- Proper greeting: 
- Complete words: 
- Professional tone: 
- Clear and specific: 
- Proper punctuation: 

**[TRANSFERRED]:** Excuse me, could you please assist me with this matter?

## Your Task

**Source Text:** "{source_text}"
**Target Style:** {target_style}

Follow the 4-step process above:"""

    return prompt


def create_cot_prompt_few_shot(source_text, target_style):
    """
    Creates a multi-step reasoning prompt for text style transfer
    """

    prompt = f"""
You are an expert at text style transfer. Your task is to transform text from one style to another. First, explain your reasoning about what needs to be changed, then provide the transformed text.
Output Format

    Provide reasoning about the style elements that need to be changed
    Mark your final transformed text with [Transferred]:

Example 1: Informal → Formal
Source Text: "hey can u help me out with this thing?" Target Style: formal

The original text is informal. The use of "hey" is a casual greeting, "u" is text speak for "you", "help me out" is casual phrasing, and "this thing" is vague. To make it formal, I need to use proper greetings, complete words, professional language, and be more specific.

[Transferred]: Excuse me, could you please assist me with this matter?


Example 2: Toxic → Non-toxic
Source Text: "you're such an idiot, this is completely wrong"
Target Style: non-toxic

The original text contains a personal insult ("you're such an idiot") and harsh language ("completely wrong"). To detoxify this, I need to remove the personal attack, soften the criticism, and focus on the issue rather than attacking the person.

[Transferred]: I don't think this approach is correct. Could we reconsider this?

Example 3: Complex → Simple
Source Text: "The implementation necessitates a comprehensive evaluation of the multifaceted parameters."
Target Style: simple

The original text uses complex vocabulary and sentence structure. "Implementation necessitates" can be simplified to "we need to", "comprehensive evaluation" to "look at", and "multifaceted parameters" to "different factors". The overall sentence structure should be more straightforward.

[Transferred]: We need to look at all the different factors.

Now transform this text:
Source Text: "{source_text}" Target Style: {target_style}


"""

    return prompt


def create_cot_prompt(source_text, target_style):
    """
    Creates a multi-step reasoning prompt for text style transfer
    """

    prompt = f"""
You are an expert at text style transfer. Your task is to transform text from one style to another. First, explain your reasoning about what needs to be changed, then provide the transformed text.
Output Format

    Provide reasoning about the style elements that need to be changed
    Preserve original meaning
    Mark your final transformed text with [Transferred]:

Source Text: "{source_text}" Target Style: {target_style}
"""

    return prompt