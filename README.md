# LLM Attention Benchmark

Standalone implementation of PI-LLM evaluation framework. Measures proactive interference and working memory limits in Large Language Models beyond context length constraints.

## Table of Contents

1. [Super Quick Start](#super-quick-start)
2. [A) Paper-Ready Results](#a-paper-ready-results-recommended-for-research)
3. [B) Automated Testing & Plotting (Beta)](#b-automated-testing--plotting-beta-)
4. [Key Features](#key-features)
5. [Available Tests](#available-tests)
6. [Advanced Usage Examples](#advanced-usage-examples)
7. [Results](#results)
8. [Folder Structure](#folder-structure)
9. [Customization](#customization)
10. [Troubleshooting](#troubleshooting)
11. [Citation](#citation)

---

## Super Quick Start

1. Install dependencies
```
pip install -r requirements.txt
```

2. Edit API.json with your actual API keys

3. Ensure your models and their corresponding test names are added to `configs/model_test_mapping.yaml`, if not already present. This is a prerequisite for utilizing the methods detailed in the subsequent section. You can assign any of the [five pre-configured tests](#available-tests) to the models you want to evaluate

4. You can [modify the parameters](#modify-test-parameters) in `configs/test*.yaml` or create your own tests using them as references.

---

## A) Paper-Ready Results (Recommended for Research)

**Exact paper implementation with CI95 statistical confidence**

Since each prompt is randomly generated, the system automatically runs enough sessions to reach CI95 statistical confidence for robust, publication-ready results.

navigate to the repository root directory (exact paper implementation)

- run one test with one model
```bash
python core/run_pi.py --test test1_updates --model your-model-name
```

- run multiple tests with one model
```bash
python run_pi.py --test test1_updates,test2_nkeys --model your-model-name
```

- If only --model is given, all tests associated with this model (in configs/model_test_mapping.yaml) will be runned
```bash
python run_pi.py --model your-model-name
```

**Note:** The `--test` flag automatically reads configuration files from the `configs/` directory (e.g., `test1_updates` → `configs/test1_updates.yaml`). 

**Benefits:**
- Exact code from published paper
- Maximum reliability and reproducibility  
- Robust statistics (CI95 confidence)
- No automation dependencies

---

## B) Automated Testing & Plotting (Beta) ⚠️

**Beta automation with flexible session control and automatic plots**

More flexible alternative that provides automated test execution, plotting, and the ability to override CI95 mode with fixed session counts for faster previews.

### Default Automation (CI95 mode):
```bash
# Automated testing with plots - uses CI95 for robust statistics
python run_pi_benchmark.py --model your-model-name
```

### Quick Preview Mode (Fixed Sessions):
```bash
# Switch to quick mode - runs fixed 5 sessions (default) instead of CI95
python run_pi_benchmark.py --model your-model-name --mode quick

# Custom session count (instead of default 5)
python run_pi_benchmark.py --model your-model-name --mode quick --n-sessions 10
```

**Benefits:**
- Beta convenience features
- Automatic plots and analysis
- Flexible session control (CI95 or fixed count)
- One command runs everything
- Good for exploration and development

**Note:** Quick mode uses fixed sessions (default: 5) for faster results but provides less robust statistics than CI95 mode.

## Key Features

- **Two Approaches**: Core paper code (A) vs Beta automation (B)
- **Zero Setup**: Test any model directly without registration or config editing
- **5 Core Tests**: Proactive interference across multiple dimensions
- **Flexible Sessions**: CI95 robust statistics or fixed count (default: 5)
- **Auto Analysis**: Evaluation → Analysis → Plots (in Beta mode)

## Available Tests

- **test1**: how retrieval accuracy degrades as interference builds by increasing updates for each tracked key. (`test1_updates`)
- **test2**: increasing the number of keys in the input (`test2_nkeys`)  
- **test3**: fix input: fix the update or each keys and the number of keys to control total input length. but varying the tracked keys in the query part to query only partial number of keys provided in the input context.  (`test3_ntracked`)
- **test4**: Item length effects on proactive interference, testing how value complexity impacts retrieval under memory pressure (`test4_itemlen`)
- **test5**: variation to test1-Sequential update presentation to assess temporal order effects on interference patterns and also serve as a extra robustness test to show interference could be introduced in various from. (`test5_updates_randomoff`)

## Advanced Usage Examples

### Multiple Tests and Custom Options

**For Core Code (A):**
```bash
# Run individual tests with core code
cd core/
python run_pi.py --config ../configs/test1_updates.yaml --model gemini-2.0-flash
python run_pi.py --config ../configs/test2_nkeys.yaml --model gemini-2.0-flash

# For quick mode in core: edit YAML file to change n_sessions from 0 to 5
# Example: edit configs/test1_updates.yaml, change "n_sessions: 0" to "n_sessions: 5"
```

**For Automation (B):**
```bash
# Single test with automation
python run_pi_benchmark.py --model gpt-4o --tests test1

# Multiple specific tests with automation
python run_pi_benchmark.py --model claude-3-5-sonnet --tests test1,test3,test5

# All 5 tests explicitly (same as default)
python run_pi_benchmark.py --model your-model --tests test1,test2,test3,test4,test5
```



## Results

Results are saved in `eval_pi/` with:
- **Test Data**: Raw test results in `test_*/your-model/`
- **Plots & CSV (Beta)**: Automated plots and summary data in `plots_and_csv_beta/`

Example structure:
```
benchmark/                              # Standalone repo root
├── eval_pi/                           # Results directory
│   ├── test1_updates/your-model/
│   │   └── Test directories with timestamps (raw data)
│   ├── test2_nkeys/your-model/
│   │   └── Test directories with timestamps (raw data)
│   └── plots_and_csv_beta/
│       ├── test1_updates_your-model/
│       │   ├── accuracy_vs_n_tracked_updates_*.png
│       │   └── accuracy_summary_*.csv
│       └── test2_nkeys_your-model/
│           ├── accuracy_vs_nkeys_*.png
│           └── accuracy_summary_*.csv
├── core/                              # Paper implementation
├── automation/                        # Beta features
└── configs/                          # Test configurations
```

## Folder Structure

```
benchmark/
├── run_pi_benchmark.py          # 🚀 Main entry point (uses automation)
├── configs/                     # ⚙️ Test configurations
│   ├── test1_updates.yaml      #    Memory update tests
│   ├── test2_nkeys.yaml        #    Key count variations
│   ├── test3_ntracked.yaml     #    Tracking ratios
│   ├── test4_itemlen.yaml      #    Item length effects
│   ├── test5_updates_randomoff.yaml  # Sequential patterns
│   └── model_test_mapping.yaml #    Optional restrictions
├── core/                        # STABLE: Paper code
│   ├── run_pi.py               #    Main test execution engine
│   ├── pi_flow_upgrade.py      #    PI flow implementation
│   ├── chat_terminal.py        #    Terminal interface
│   └── analysis_helper.py      #    Core analysis utilities
├── automation/ ⚠️               # 🚧 BETA: Quick preview
│   ├── run_full_pipeline.py    #    Automation pipeline
│   ├── analyze_pi_custom.py    #    Custom analysis
│   ├── plot_accuracy_trend.py  #    Plotting engine
│   └── analyze_pi_flow_final.py #   Advanced workflows
├── testing_data/               # 📊 Test datasets
└── requirements.txt            # 📦 Dependencies
```

### Code Stability Guide:
- **core/**: ✅ Stable paper code - use for research
- **automation/**: ⚠️ Beta features - use for quick preview

## 🔧 Customization

### Modify Test Parameters
Edit files in `configs/`:
- Change `n_tracked_updates` arrays for different test points
- Adjust `max_tokens`, `temperature` for model behavior
- Modify `source_dict_path` for different datasets

### Add New Tests
1. Create new `.yaml` in `configs/`
2. Add test name to `DEFAULT_TESTS` in `run_pi_benchmark.py`
3. Run: `python run_pi_benchmark.py --model your-model --tests your-new-test`

### Model Mapping (Optional - Batch Runs Only)
**This file is completely OPTIONAL and NOT required for normal use.**

**Purpose**: Only for massive batch runs where you want to automatically assign different test settings to different models.

**When to use**: 
- You have many models to test
- Different models should run different test suites (e.g., expensive models run fewer tests)
- You want to automate which tests each model runs

**How it works**:
1. Edit `configs/model_test_mapping.yaml` to assign test suites:
```yaml
expensive-model:
  - test1_updates        # Light testing only
cheap-model:
  - test1_updates
  - test2_nkeys
  - test3_ntracked       # Full testing
```

2. Use the mapping: `python run_pi_benchmark.py --model your-model --use-mapping`

**Normal usage**: Just run any model directly without this file - `python run_pi_benchmark.py --model any-model-name`

## 🛟 Troubleshooting

**Q: "Test configuration not found"**
```bash
# Check if config exists
ls configs/test*.yaml
```

**Q: "Command failed"**
```bash
# Check dependencies
pip install -r requirements.txt

# Check if test data exists
ls testing_data/dict_category_double-word_46-400_v1-1.json
```

**Q: "No results generated"**
```bash
# Check eval_pi directory
ls eval_pi/

# Look for error logs
find eval_pi/ -name "*.log" -o -name "*terminal_log*"
```


## Citation

This benchmark implements the PI-LLM evaluation framework from:

**"Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length"**  
*Chupei Wang, Jiaqiu Vince Sun*  
arXiv:2506.08184 [cs.CL]  
[Paper Link](https://arxiv.org/abs/2506.08184)

```bibtex
@article{wang2025unable,
  title={Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length},
  author={Wang, Chupei and Sun, Jiaqiu Vince},
  journal={arXiv preprint arXiv:2506.08184},
  year={2025}
}
```

