"""Document classification: what role a file plays, distinct from how to chunk it."""

from workspace_indexer.classification.classification import Classification
from workspace_indexer.classification.document_classifier import DocumentClassifier
from workspace_indexer.classification.frontmatter_rule import FrontmatterRule
from workspace_indexer.classification.modal_density_rule import ModalDensityRule, modal_density
from workspace_indexer.classification.path_rule import PathRule
from workspace_indexer.classification.rule import Rule
from workspace_indexer.classification.rule_classifier import RuleClassifier

__all__ = [
    "Classification",
    "DocumentClassifier",
    "FrontmatterRule",
    "ModalDensityRule",
    "PathRule",
    "Rule",
    "RuleClassifier",
    "modal_density",
]
