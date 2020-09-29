# Refactor

1. HTML files for the following should be separated.
	* Landing page
	* Prospective Power Analysis
	* Data Analysis
	* Significance Testing
	* Effect Size Estimation
	* Retrospective Power Analysis
	* Download Report
	
2. The `About` modal should be replaced by the landing page and removed. The `Save Settings` modal should be replaced by a save button next to the run buttons.

3. Code changes! Tags like `<p>`, `<abbr>`, `<section>` should replace our manual breaking everywhere. IDs and variable names should all be reguularized and predictable.

4. CSS should be rewritten to be simpler. CSS should be moved to stylesheets, except where changeable. JS should be moved to two files, one for interface maintence and state-checking, and one for all form validation. JS should be simplified so that functions are kept to a minimum, reused, and composed. Tooltip and modal files should all go.

