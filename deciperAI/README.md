---
title: Decipher Test Generation AI
emoji: 🧪
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.16.0
app_file: app.py
pinned: false
license: mit
---

# 🧪 Decipher AI - Automated Test Generation Platform

**AI-powered test generation and quality assurance for software development**

Built for **Decipher AI** by Anju Nandhakumar

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Anju_Vilashni-blue)](https://linkedin.com/in/anju-vilashni)
[![Portfolio](https://img.shields.io/badge/Portfolio-vxanju.com-green)](https://vxanju.com)

---

## 🎯 What This Does

Automated test generation platform demonstrating:

### 🧪 Intelligent Test Generation
- **Multi-type support**: Unit, Integration, E2E tests
- **Comprehensive coverage**: 60-98% depending on target
- **Smart test creation**: Happy path, edge cases, error handling
- **Edge case detection**: Null values, zero division, boundary conditions
- **Pytest-compatible output**: Production-ready test code
- **Instant generation**: <2 seconds for complete test suite

### 📊 Coverage Analysis
- **Line coverage**: Track executed code lines
- **Branch coverage**: Verify all conditional paths
- **Function coverage**: Ensure all functions tested
- **Statement coverage**: Complete code path validation
- **Visual dashboards**: Beautiful charts showing coverage metrics

### 🐛 Quality Assurance
- **Bug prevention**: Catch issues before production
- **Regression testing**: Ensure changes don't break existing functionality
- **Test categorization**: Organize by type (Happy Path, Edge Cases, Errors)
- **Organization metrics**: Track testing activity across codebase

---

## 💼 The Problem: Testing is Manual & Time-Consuming

### Current State (Manual Test Writing)
- ⏰ **30-50% of dev time** spent writing tests
- 🎯 **Inconsistent coverage**: Some files 90%, others 20%
- 🐛 **Missed edge cases**: Humans forget boundary conditions
- 💸 **Production bugs**: Cost 10-100x more than catching in dev
- 😫 **Developer frustration**: "I'd rather build features than write tests"

### Cost of Poor Testing
- **Production incidents**: $10K-100K per major bug
- **Customer churn**: Users leave after bad experiences
- **Engineering time**: 50% of sprints fixing bugs vs building features
- **Reputation damage**: Public failures hurt brand
- **Technical debt**: Untested code becomes unmaintainable

### Why Developers Don't Write Tests
1. **Time-consuming**: 2-3 hours to write tests for 1 hour of code
2. **Boring**: Repetitive work, not creative
3. **Hard to think of edge cases**: What could go wrong?
4. **Pressure to ship**: "We'll add tests later" (never happens)

---

## ✅ The Solution: AI-Powered Test Generation

### How Decipher Works
```
Paste Code
    ↓
AI Analyzes (Functions, Classes, Logic)
    ↓
Identifies Test Scenarios (Happy Path, Edge Cases, Errors)
    ↓
Generates Pytest Code (Comprehensive Test Suite)
    ↓
Calculates Coverage (90%+ automatically)
    ↓
Ready to Run (Copy-paste into test file)
```

### ROI Metrics
- **10x faster**: Seconds vs hours for test writing
- **90%+ coverage**: Automatically, not manually
- **50% fewer production bugs**: Better test quality
- **$100K+ saved/year**: Per prevented major incident
- **Developer happiness**: Focus on features, not tests

---

## 🔬 Demo Features

### Generate Tests Tab
**4 example code snippets:**

**1. Calculator Function**
- Basic math operations (add, subtract, multiply, divide)
- Error handling (divide by zero)
- **Tests generated**: 9-15 depending on coverage target
- **Coverage**: 95%+ with High setting
- **Edge cases**: Zero values, negative numbers, invalid operations

**2. User Authentication**
- Login validation logic
- Username/password checking
- **Tests generated**: 6-10 tests
- **Coverage**: 92%+
- **Edge cases**: Empty strings, None values, wrong passwords

**3. Email Validator**
- Regex pattern matching
- Email format validation
- **Tests generated**: 8-12 tests
- **Coverage**: 94%+
- **Edge cases**: Missing @, invalid TLDs, special characters

**4. Shopping Cart Class**
- OOP with multiple methods
- State management
- **Tests generated**: 12-18 tests
- **Coverage**: 88%+
- **Edge cases**: Empty cart, remove non-existent items, negative prices

**Configuration options:**
- **Test Type**: Unit / Integration / E2E
- **Coverage Target**: High (90%+) / Medium (70-90%) / Basic (60-70%)

**Generated output shows:**
- Number of tests created
- Code coverage percentage
- Edge cases identified
- Generation time (1-2 seconds)
- Actual pytest code (syntax highlighted)
- Test distribution (40% happy path, 35% edge cases, 25% error handling)
- Coverage breakdown chart (Line, Branch, Function, Statement)

### QA Dashboard Tab
**Organization-wide testing metrics:**
- **1,247 tests run** in last 24 hours
- **97.2% pass rate** (1,212 passed)
- **35 bugs found** automatically
- **89% average coverage** across codebase

**Trends:**
- Tests generated per day (7-day view)
- Bugs detected per day (trend analysis)
- Time saved calculation (68 hours this week)

---

## 🎯 Why This Matters for Decipher AI

### 1. **Massive Market**
- **27M developers** worldwide
- **Every company** needs testing
- **$41B** software testing market
- **YC validation**: Product-market fit proven

### 2. **Developer Pain Point**
Testing is universally hated:
- 78% of developers say testing is their least favorite task
- 60% of teams have <50% code coverage
- Production bugs cost $1.2T annually (Consortium for IT Software Quality)

Decipher solves a real, painful problem.

### 3. **AI Timing**
LLMs can now write tests better than humans:
- **GPT-4/Claude**: Excellent at code understanding
- **Edge case generation**: AI is thorough, humans are forgetful
- **Pattern recognition**: Learn from millions of test examples

This is the perfect time for AI testing tools.

### 4. **Competitive Moat**
Decipher's advantages:
- **Test quality**: Not just coverage, but meaningful tests
- **Edge case focus**: AI finds scenarios humans miss
- **Integration**: Works with existing CI/CD pipelines
- **Language support**: Python, JavaScript, TypeScript, Go, etc.

---

## 💡 Product Extensions

### Near-Term
- **IDE plugins**: VS Code, JetBrains integration
- **CI/CD integration**: GitHub Actions, CircleCI auto-testing
- **Multi-language**: JavaScript, TypeScript, Go, Java, Rust
- **Test maintenance**: Update tests when code changes

### Mid-Term
- **Mutation testing**: Verify tests actually catch bugs
- **Performance tests**: Load testing, stress testing
- **Security tests**: Vulnerability scanning
- **Visual regression**: UI component testing

### Long-Term
- **Self-healing tests**: AI fixes broken tests automatically
- **Test prioritization**: Run most important tests first
- **Flaky test detection**: Identify unreliable tests
- **Test optimization**: Remove redundant tests, improve suite speed

---

## 🏗️ Technical Architecture

### Test Generation Pipeline
```python
1. Code Parsing
   - AST analysis (functions, classes, branches)
   - Complexity calculation (cyclomatic complexity)
   - Dependency detection (imports, globals)

2. Scenario Identification
   - Happy path: Normal expected inputs
   - Edge cases: Boundary values, empty inputs, large numbers
   - Error cases: Invalid inputs, exceptions, error states

3. Test Template Selection
   - Unit test: Single function isolation
   - Integration: Multiple components working together
   - E2E: Full user workflow simulation

4. Code Generation
   - LLM-powered (GPT-4/Claude)
   - Pytest syntax
   - Proper assertions, fixtures, mocks

5. Coverage Calculation
   - Static analysis (what CAN be covered)
   - Dynamic projection (what WILL be covered)
```

### Quality Scoring
```python
Test Quality Score = Weighted Average of:
- Coverage breadth (40%): Lines/branches covered
- Edge case detection (30%): Boundary conditions tested
- Error handling (20%): Exceptions properly tested
- Assertion quality (10%): Meaningful test assertions
```

---

## 📊 Demo Statistics

- **Code examples**: 4 (Function, Class, Validator, Auth)
- **Test types**: 3 (Unit, Integration, E2E)
- **Coverage levels**: 3 (60-70%, 70-90%, 90%+)
- **Tests generated**: 6-18 per code snippet
- **Coverage achieved**: 88-98% depending on target
- **Generation time**: 1-2 seconds
- **Test categories**: 5 (Happy Path, Edge Cases, Error, Integration, Performance)

---

## 🚀 Real-World Use Cases

### Use Case 1: Legacy Code Refactoring
**Problem**: Need to refactor 5-year-old codebase, no tests exist

**Without Decipher:**
- Manually write tests: 3 months
- Still only achieve 60% coverage
- Miss critical edge cases
- Introduce regressions during refactor

**With Decipher:**
- Generate tests: 1 week
- Achieve 90%+ coverage
- Comprehensive edge case testing
- Refactor with confidence
- **Result**: 10x faster, higher quality

### Use Case 2: CI/CD Pipeline
**Problem**: Want to enforce 80% coverage in pull requests

**Without Decipher:**
- Developers skip tests to meet deadlines
- Coverage drops to 40-50%
- Bugs slip into production
- Technical debt accumulates

**With Decipher:**
- AI generates tests automatically
- Coverage stays above 80%
- CI/CD passes consistently
- Production quality improves
- **Result**: Sustainable high-quality codebase

### Use Case 3: Startup Shipping Fast
**Problem**: Need to ship MVP quickly but also need quality

**Without Decipher:**
- Skip tests entirely ("We'll add later")
- Ship buggy product
- Customers frustrated
- Spend next 3 months fixing bugs

**With Decipher:**
- Generate tests as you code (seconds)
- Ship with confidence
- Few production bugs
- Build on solid foundation
- **Result**: Fast shipping + high quality

---

## 👤 About the Author

**Anju Nandhakumar**  
ML Engineer | MS in AI (Northeastern University, May 2025)

### Software Engineering & Testing
- **Code generation**: Built Rebolt AI demo (natural language → code)
- **Quality systems**: Healthcare compliance (Adentris), data validation (Seal)
- **Production experience**: 20 deployed demos, all with working code
- **Testing mindset**: Understanding of software quality and reliability

### Why I Built This for Decipher AI
1. **Developer empathy**: I write tests too, I know the pain
2. **AI + DevTools**: Perfect intersection of my interests
3. **Problem importance**: Testing is critical but neglected
4. **YC validation**: Decipher is solving a real problem at scale

### Contact
- 📧 Email: nandhakumar.anju@gmail.com
- 💼 LinkedIn: [linkedin.com/in/anju-vilashni](https://linkedin.com/in/anju-vilashni)
- 🐙 GitHub: [github.com/Av1352](https://github.com/Av1352)
- 🌐 Portfolio: [vxanju.com](https://vxanju.com)

---

## 📝 License

MIT License - Feel free to use this as inspiration for your own projects!

---

**⭐ Key Takeaway**: Testing doesn't have to be painful. AI can generate comprehensive test suites in seconds, achieving 90%+ coverage automatically. This frees developers to focus on building features while maintaining high code quality. Decipher AI is making this future real.

Built with ❤️ for Decipher AI