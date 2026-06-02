{{/*
Common labels for all resources
*/}}
{{- define "prompt-optimizer.labels" -}}
app: prompt-optimizer
app.kubernetes.io/name: prompt-optimizer
app.kubernetes.io/part-of: lang-learn
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{/*
Selector labels (used in Service → Deployment matching)
*/}}
{{- define "prompt-optimizer.selectorLabels" -}}
app: prompt-optimizer
{{- end }}
