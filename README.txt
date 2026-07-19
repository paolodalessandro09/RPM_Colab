RPM refactor base package
=========================

This folder is the new base version of the RPM/RFM scripts.

Core files:
  rpm_refac.py          RPM and RPMLayer classes, including agop, gradient_descent, and foof metric-update paths.
  train_rpm.py          Unified training script. Controlled by config files.
  speed_refac.py        Mahalanobis kernels and TorchSpecNystrom feature construction.
  analyze_rpm_run.py    Post-run analysis utilities.
  config_refac.py       Flexible YAML config loader.
  logger_refac.py       Experiment logger.
  util_refac.py         Metrics, PSD projection, Fisher/effective-dimension helpers.
  dspTools.py           Data/signal utilities.

Configs:
  config_agop.yaml
  config_gradient_fixed_readout.yaml
  config_foof_fixed_readout.yaml

All three configs use one RPM layer, kernel_sigma: [0.5], and normalize_features: false.

Run:
  python train_rpm.py config_agop.yaml
  python train_rpm.py config_gradient_fixed_readout.yaml
  python train_rpm.py config_foof_fixed_readout.yaml
  
This will run the training script with the config file (.yaml) that specifies the run parameters. 



