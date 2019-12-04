## Monitor the system stats with Prometheus
**Refer** - https://en.wikipedia.org/wiki/Prometheus_(software)
1. Download the binary of `prometheus`
```bash
cd ~/Downloads
wget https://github.com/prometheus/prometheus/releases/download/v2.14.0/prometheus-2.14.0.linux-amd64.tar.gz
tar -xvf prometheus-2.14.0.linux-amd64.tar.gz
```
2. To run the prometheus
```bash
cd ~/Downloads/prometheus-2.14.0.linux-amd64
./prometheus
```
3. For GPU stats
```bash
cd ~/Downloads/prometheus-2.14.0.linux-amd64
git clone https://github.com/mindprince/nvidia_gpu_prometheus_exporter.git
./nvidia_gpu_prometheus_exporter
```
4. Configure the prometheus to scrape the nvidia gpu metrics
```bash
cd ~/Downloads/prometheus-2.14.0.linux-amd64
```
```json
# my global config
global:
  scrape_interval:     15s # Set the scrape interval to every 15 seconds. Default is every 1 minute.
  evaluation_interval: 15s # Evaluate rules every 15 seconds. The default is every 1 minute.
  # scrape_timeout is set to the global default (10s).

# Alertmanager configuration
alerting:
  alertmanagers:
  - static_configs:
    - targets:
      # - alertmanager:9093

# Load rules once and periodically evaluate them according to the global 'evaluation_interval'.
rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

# A scrape configuration containing exactly one endpoint to scrape:
# Here it's Prometheus itself.
scrape_configs:
  # The job name is added as a label `job=<job_name>` to any timeseries scraped from this config.
  - job_name: 'prometheus'

    # metrics_path defaults to '/metrics'
    # scheme defaults to 'http'.

    static_configs:
    - targets: ['localhost:9090']
  - job_name: 'nvidia_gpu_prometheus_exporter'

    # metrics_path defaults to '/metrics'
    # scheme defaults to 'http'.

    static_configs:
    - targets: ['localhost:9445']

```
5. To visualize the metrics, download Grafana
```bash
cd ~/Downloads/prometheus-2.14.0.linux-amd64
wget https://dl.grafana.com/oss/release/grafana-6.5.0.linux-amd64.tar.gz
tar -zxvf grafana-6.5.0.linux-amd64.tar.gz
cd ~/Downloads/prometheus-2.14.0.linux-amd64/grafana-6.5.0.linux-amd64.tar.gz
./bin/grafana-server
```
* Add the prometheus datasource in the grafana
* Import the dashboard from searching on google i.e. copy paste the number

##References - 
* https://prometheus.io/download/
* https://github.com/mindprince/nvidia_gpu_prometheus_exporter
* https://grafana.com/grafana/download
* https://github.com/prometheus/node_exporter
* https://github.com/netdata/netdata
* Video tutorial - https://www.youtube.com/watch?v=4WWW2ZLEg74