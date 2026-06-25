{{/*
Common labels for all MLflow resources
*/}}
{{- define "mlflow.labels" -}}
app.kubernetes.io/part-of: lang-learn
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{/*
PostgreSQL selector labels
*/}}
{{- define "mlflow.postgres.selectorLabels" -}}
app: mlflow-postgres
{{- end }}

{{/*
MLflow server selector labels
*/}}
{{- define "mlflow.server.selectorLabels" -}}
app: mlflow-tracking
{{- end }}
