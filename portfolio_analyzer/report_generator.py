from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import logging

# Determine the correct path to the templates directory
# This assumes 'report_generator.py' is in 'portfolio_analyzer/'
# and 'templates/' is in the parent directory of 'portfolio_analyzer/'.
TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


def generate_html_report(report_sections: dict, output_path: str = "portfolio_report.html",
                         drilldown_data: str = "[]"):
  """
  Generates an HTML report from a dictionary of sections using a template file.
  Each value in report_sections should be a dictionary with 'type' and 'html' or 'charts' (list of HTML strings).
  """
  if not TEMPLATE_DIR.exists():
    logging.error(f"Template directory not found: {TEMPLATE_DIR}")
    return
  if not (TEMPLATE_DIR / "report_template.html").exists():
    logging.error(f"Report template file 'report_template.html' not found in: {TEMPLATE_DIR}")
    return

  try:
    template_loader = FileSystemLoader(searchpath=str(TEMPLATE_DIR))  # Ensure searchpath is a string
    template_env = Environment(loader=template_loader)
    template = template_env.get_template("report_template.html")
  except Exception as e:
    logging.error(f"Error loading Jinja2 template: {e}")
    return

  try:
    html_content = template.render(sections=report_sections, drilldown_data=drilldown_data)
  except Exception as e:
    logging.error(f"Error rendering Jinja2 template: {e}")
    # Optionally print report_sections structure for debugging
    # import json
    # logging.debug(f"Report sections data: {json.dumps(report_sections, indent=2, default=str)}")
    return

  try:
    with open(output_path, "w", encoding="utf-8") as f:
      f.write(html_content)
    logging.info(f"Report generated: {output_path}")
  except Exception as e:
    logging.error(f"Error writing HTML report to file: {e}")
