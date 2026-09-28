# Responsible Use

This repository is a research artifact for studying emergent misalignment and
parameter-space interventions in language models. Its intended uses are
scientific replication, mechanism auditing, and safety research.

## Content

The frozen data contain model-generated examples of unsafe behavior, including
dangerous financial, medical, and physical-risk advice. These records are
included because they are measurement inputs to the reported analyses. They are
not recommendations or instructions for real-world action.

The released adapters include controlled harmful-direction amplification
conditions as well as ablations and random-direction controls. Amplification
adapters are experimental measurement artifacts. Their inclusion enables audit
of the reported direction-specific effects and does not indicate that they are
suitable for deployment.

## Use and Reporting

Users should preserve the distinction between fixed-response measurements and
free-generation behavior. Fixed-response scores measure conditional support for
frozen text and are not estimates of an intervention's free-generation EM rate.

When reporting results derived from this repository, identify the model,
dataset, checkpoint interval, basis checkpoint, parameter block, intervention,
and readout. Do not generalize a representative released configuration to models
or datasets that were not evaluated.

Users remain responsible for following the base-model terms and for evaluating
the risks of any model or adapter deployment.
