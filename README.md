# DataLab

DataLab is a Streamlit-based data analysis app for loading datasets, inspecting them, cleaning data, visualizing results, and collecting basic machine-learning task requirements.

The project is organized as a small multi-page application with a shared dataset state. A file loaded on the Home page is available to the other pages through the same Streamlit session.

## What the app does

- Load CSV, Excel, and JSON files from the sidebar
- Load a bundled sample dataset for quick testing
- Show a dataset summary on the Home page
- Provide placeholder pages for data cleaning and visualization
- Collect ML task details and suggest a model shortlist on the ML Lab page

## Tech Stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| UI | Streamlit | Multi-page web interface and sidebar controls |
| Data handling | pandas | Dataset loading, previewing, and summary metrics |
| Numerical support | NumPy | Basic numerical operations |
| Excel support | openpyxl, xlrd | Reading `.xlsx` and `.xls` files |

## Current Project Structure

```text
datalab/
	src/
		main.py
		testy.csv
		core/
			dataset_controller.py
		data/
			data_loader.py
			data_cleaner.py
			pipeline.py
		gui/
			pages/
				main_window.py
				data_visualization.py
				data_cleaning.py
				ML_Lab.py
```

## Main Pages

- **Home**: shows the active dataset summary and preview
- **Data Visualization**: Displays descriptive data analytics, custom data plotting
- **Data Cleaning**: placeholder page for cleaning actions
- **ML Lab**: asks questions about the task and data, then suggests three model options

## Dataset Flow

1. Load a dataset from the global sidebar or use the bundled sample dataset.
2. The dataset is stored in `st.session_state.dataset_controller`.
3. The Home page displays the current shape, column list, missing values, and preview.
4. The ML Lab page reads the same dataset state to auto-fill size-related questions when data is already available.

## What I've learned

- From the basic Streamlit to some advanced features
- Designing a webpage and a backend behind
- Efficient Github Copilot usage than my previous project, this sped up the development by about 250%
- How to avoid Vibe Coding, by using Copilot to fill my knowledge gap, not solve my problems

## How AI was used

- For the initial stages of planning and featuees
- For repetitive task, such as Selectbox options, or implementing a menu pattern several times
- Potentional flawed / redundant code checks, debugging
- Along the development, any quick questions about Streamlit

## How to Run

From the `datalab/src` directory:

```bash
streamlit run main.py
```

## Dependencies

Install the packages listed in `datalab/src/requirements.txt`:

```bash
pip install -r requirements.txt
```

## Notes

- The current cleaning page is still minimal.
- The ML Lab recommendation logic is rule-based and uses form answers plus dataset size hints.
- The app currently focuses on local file loading and interactive exploration rather than a full production pipeline.
