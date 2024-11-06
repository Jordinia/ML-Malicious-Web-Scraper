# ML-Malicious-Web-Scraper

## Overview

This project involves developing a Website Classifier using machine learning, specifically leveraging Large Language Models (LLMs). The classifier is designed to categorize websites based on their content, identifying potentially malicious or benign sites.

## Project Structure

```
Labeling/
	ai_labelling_v2.ipynb
    ai_labelling.ipynb
	domain_classification.log
	logs/
		done/
	prompts/
Scraper/
	1_download_commoncrawl.py
	2_warc_scraper_fastwarc_single.py
	2_warc_scraper_fastwarc.py
	2_warc_scraper_warcio_single.py
	3_extract_unique_domains_csv.py
	4_filter_duplicates.py
	5_add_label.py
	csv_row_split.py
	phishstorm.ipynb
	row_checker.ipynb
	scraper.ipynb
	wordfinder.py
	wsl_selenium_install.sh
cc-index.paths
wet.paths
README.md
```

## Notebooks

- `Labeling/ai_labelling.ipynb`: Contains the main logic for labeling websites using LLMs.
- `Labeling/ai_labelling_v2.ipynb`: An updated version of the labeling notebook with logging feature.

## Scripts

- `Scraper/1_download_commoncrawl.py`: Downloads data from Common Crawl.
- `Scraper/2_warc_scraper_fastwarc_single.py`: Scrapes WARC files using the FastWARC library.
- `Scraper/2_warc_scraper_fastwarc.py`: Another version of the WARC scraper using FastWARC.
- `Scraper/2_warc_scraper_warcio_single.py`: Scrapes WARC files using the WarcIO library.
- `Scraper/3_extract_unique_domains_csv.py`: Extracts unique domains from the scraped data.
- `Scraper/4_filter_duplicates.py`: Filters duplicate entries from the data.
- `Scraper/5_add_label.py`: Adds labels to the data.
- `Scraper/csv_row_split.py`: Splits CSV rows for processing.
- `Scraper/wordfinder.py`: Finds specific words in the data.
- `Scraper/wsl_selenium_install.sh`: Installs Selenium on WSL.

## Logs

- `Labeling/logs/`: Contains logs of the domain classification process.

## Prompts

- `Labeling/prompts/`: Contains prompt files used for the LLM.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/ML-Malicious-Web-Scraper.git
    ```
2. Navigate to the project directory:
    ```sh
    cd ML-Malicious-Web-Scraper
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the data download script:
    ```sh
    python Scraper/1_download_commoncrawl.py
    ```
2. Scrape the WARC files:
    ```sh
    python Scraper/2_warc_scraper_fastwarc_single.py
    ```
3. Extract unique domains:
    ```sh
    python Scraper/3_extract_unique_domains_csv.py
    ```
4. Filter duplicates:
    ```sh
    python Scraper/4_filter_duplicates.py
    ```
5. Add labels to the data:
    ```sh
    python Scraper/5_add_label.py
    ```

## License

This project is licensed under the MIT License.