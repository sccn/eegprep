# Documentation Style Guide

This guide establishes writing standards, formatting conventions, and best practices for eegprep documentation.

## Table of Contents

- [Writing Style](#writing-style)
- [Tone and Voice](#tone-and-voice)
- [Formatting Conventions](#formatting-conventions)
- [Code Examples](#code-examples)
- [Terminology](#terminology)
- [Documentation Checklist](#documentation-checklist)

## Writing Style

### General Principles

1. **Clarity First**: Write for clarity, not cleverness
2. **Active Voice**: Prefer active voice over passive voice
3. **Conciseness**: Be concise without sacrificing clarity
4. **Consistency**: Use consistent terminology and formatting
5. **Accessibility**: Write for users of varying skill levels

### Sentence Structure

**Good**:
- Use short, simple sentences
- One idea per sentence
- Avoid nested clauses

**Bad**:
- Long, complex sentences with multiple clauses
- Multiple ideas in one sentence

**Example**:
```
Good: The filter removes noise from the signal. It uses a Butterworth design.
Bad: The filter, which uses a Butterworth design, removes noise from the signal.
```

### Paragraph Structure

- Start with a topic sentence
- Keep paragraphs short (3-5 sentences)
- Use transitions between paragraphs
- End with a conclusion or transition

### Headings

- Use descriptive, specific headings
- Use sentence case (capitalize first word only)
- Avoid generic headings like "Overview"
- Use consistent heading hierarchy

**Good**:
```
## Preprocessing pipeline overview
### Filtering and artifact removal
### ICA decomposition
```

**Bad**:
```
## Overview
### Details
### More Details
```

## Tone and Voice

### Professional but Approachable

- Write as if explaining to a colleague
- Avoid overly formal language
- Avoid slang and colloquialisms
- Be respectful and inclusive

### Audience Awareness

**For Beginners**:
- Explain concepts before using them
- Provide more examples
- Link to background material

**For Advanced Users**:
- Assume familiarity with concepts
- Focus on implementation details
- Provide references for deeper learning

### Positive Language

- Focus on what users can do
- Avoid negative phrasing

**Good**: "You can improve accuracy by..."
**Bad**: "Don't use this method because it's inaccurate..."

### Inclusive Language

- Use "they/them" for singular pronouns
- Avoid gendered language
- Use "user" instead of "he/she"
- Avoid ableist language

## Formatting Conventions

### Markdown Formatting

**Bold**: Use for emphasis on important terms
```markdown
The **preprocessing pipeline** consists of several stages.
```

**Italics**: Use for variable names, file names, and emphasis
```markdown
The *sampling_rate* parameter controls the output frequency.
```

**Code**: Use for code, commands, and technical terms
```markdown
Use the `clean_artifacts()` function to remove artifacts.
```

**Links**: Use descriptive link text
```markdown
See the [preprocessing guide](user_guide/preprocessing.rst) for details.
```

### Lists

**Ordered Lists**: Use for sequential steps
```markdown
1. Load the EEG data
2. Apply preprocessing filters
3. Perform ICA decomposition
```

**Unordered Lists**: Use for non-sequential items
```markdown
- Filtering
- Artifact removal
- ICA decomposition
```

**Definition Lists**: Use for term definitions
```markdown
Preprocessing
: The process of preparing raw EEG data for analysis

Artifact
: Unwanted signals in the EEG data
```

### Tables

Use tables for structured information:

```markdown
| Method | Pros | Cons |
|--------|------|------|
| ASR | Fast | May remove valid data |
| ICA | Flexible | Requires manual inspection |
```

### Admonitions

Use for special information:

```rst
.. note::
   This is important information.

.. warning::
   Be careful with this parameter.

.. tip::
   This is a helpful suggestion.

.. seealso::
   See also the related function.
```

### Code Blocks

Specify language for syntax highlighting:

````markdown
```python
from eegprep import clean_artifacts

# Load data
eeg = load_eeg_data('data.set')

# Clean artifacts
eeg_clean = clean_artifacts(eeg)
```
````

## Code Examples

### Example Structure

1. **Brief description**: What the example does
2. **Imports**: Required imports
3. **Data loading**: Load or create sample data
4. **Processing**: Main processing steps
5. **Visualization**: Show results (optional)
6. **Explanation**: Explain key points

### Example Template

```python
"""
Example: Preprocessing EEG data with artifact removal

This example demonstrates how to load EEG data and remove artifacts
using the ASR (Artifact Subspace Reconstruction) method.
"""

# Import required libraries
import numpy as np
from eegprep import load_eeg_data, clean_artifacts

# Load sample EEG data
eeg = load_eeg_data('sample_data.set')
print(f"Loaded EEG data: {eeg.shape}")

# Remove artifacts using ASR
eeg_clean = clean_artifacts(eeg, method='asr')
print(f"Cleaned EEG data: {eeg_clean.shape}")

# Visualize results
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.plot(eeg[0, :1000], label='Original')
plt.plot(eeg_clean[0, :1000], label='Cleaned')
plt.legend()
plt.show()
```

### Code Example Guidelines

- **Runnable**: Examples should execute without errors
- **Self-contained**: Include all necessary imports and data
- **Commented**: Explain non-obvious steps
- **Realistic**: Use realistic data and parameters
- **Concise**: Keep examples focused and brief

### Docstring Examples

Use doctest format in docstrings:

```python
def clean_artifacts(eeg, method='asr'):
    """
    Remove artifacts from EEG data.
    
    Parameters
    ----------
    eeg : ndarray
        EEG data (channels x samples)
    method : str
        Artifact removal method ('asr' or 'ica')
    
    Returns
    -------
    eeg_clean : ndarray
        Cleaned EEG data
    
    Examples
    --------
    >>> eeg = np.random.randn(32, 1000)
    >>> eeg_clean = clean_artifacts(eeg)
    >>> eeg_clean.shape
    (32, 1000)
    """
```

## Terminology

### Consistent Terminology

Maintain a glossary of key terms and use them consistently:

| Term | Definition | Usage |
|------|-----------|-------|
| EEG | Electroencephalography | Use "EEG" not "electroencephalography" |
| Artifact | Unwanted signals | Use "artifact" not "noise" or "contamination" |
| ICA | Independent Component Analysis | Use "ICA" not "independent components" |
| Preprocessing | Data preparation | Use "preprocessing" not "pre-processing" |

### Abbreviations

- Define abbreviations on first use
- Use consistent abbreviations throughout
- Avoid unnecessary abbreviations

**Good**: "Independent Component Analysis (ICA) is used to decompose signals. ICA can identify artifacts."

**Bad**: "ICA (Independent Component Analysis) is used. IC (Independent Components) are identified."

### Technical Terms

- Explain technical terms for general audience
- Use consistent terminology
- Link to glossary for definitions

**Good**: "The preprocessing pipeline uses Independent Component Analysis (ICA), a technique that separates mixed signals into independent components."

**Bad**: "The pipeline uses ICA."

### Capitalization

- Capitalize proper nouns (MNE, EEGLAB, etc.)
- Capitalize acronyms (ICA, ASR, etc.)
- Use lowercase for common terms (preprocessing, artifact, etc.)

## Documentation Checklist

### Before Writing

- [ ] Understand the topic thoroughly
- [ ] Identify the target audience
- [ ] Outline the main points
- [ ] Gather examples and references

### While Writing

- [ ] Use clear, simple language
- [ ] Follow the style guide
- [ ] Use consistent terminology
- [ ] Include relevant examples
- [ ] Add cross-references
- [ ] Use proper formatting

### Before Publishing

- [ ] Proofread for grammar and spelling
- [ ] Check for consistency
- [ ] Verify all links work
- [ ] Test all code examples
- [ ] Review for clarity
- [ ] Check formatting
- [ ] Verify images and diagrams
- [ ] Get peer review

### Content Checklist

- [ ] Title is descriptive
- [ ] Introduction explains purpose
- [ ] Content is well-organized
- [ ] Examples are provided
- [ ] Key points are highlighted
- [ ] Related topics are linked
- [ ] Conclusion summarizes main points
- [ ] References are provided

### Code Example Checklist

- [ ] Example is runnable
- [ ] All imports are included
- [ ] Data loading is shown
- [ ] Output is explained
- [ ] Comments explain key steps
- [ ] Example is realistic
- [ ] Example is concise
- [ ] Example follows style guide

### API Documentation Checklist

- [ ] Function/class name is clear
- [ ] Purpose is explained
- [ ] Parameters are documented
- [ ] Return values are documented
- [ ] Exceptions are documented
- [ ] Examples are provided
- [ ] Related functions are linked
- [ ] Notes and warnings are included

## Common Mistakes to Avoid

### Writing Mistakes

- **Passive voice**: "The data was processed" → "We processed the data"
- **Vague pronouns**: "It can be used" → "The function can be used"
- **Jargon without explanation**: Explain technical terms
- **Inconsistent terminology**: Use the same term consistently
- **Too much information**: Focus on what's important

### Formatting Mistakes

- **Inconsistent heading levels**: Use proper hierarchy
- **Missing code syntax highlighting**: Specify language
- **Broken links**: Test all links
- **Inconsistent spacing**: Use consistent formatting
- **Poor table formatting**: Align columns properly

### Example Mistakes

- **Non-runnable examples**: Test all examples
- **Missing imports**: Include all necessary imports
- **Unrealistic data**: Use realistic examples
- **Unexplained steps**: Comment non-obvious code
- **Too long**: Keep examples focused

## Tools and Resources

### Writing Tools

- **Grammarly**: Grammar and spell checking
- **Hemingway Editor**: Readability analysis
- **Vale**: Documentation linting

### Markdown Tools

- **Markdown Preview**: Preview formatting
- **Markdown Linter**: Check formatting
- **Link Checker**: Verify links

### Code Tools

- **Pylint**: Code quality
- **Black**: Code formatting
- **Pytest**: Test examples

## References

- [Google Style Guide](https://google.github.io/styleguide/docguide/)
- [Microsoft Writing Style Guide](https://docs.microsoft.com/en-us/style-guide/welcome/)
- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
