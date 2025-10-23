{{/*
Convert models list to comma-separated string for environment variable
*/}}
{{- define "llm-engine.modelsString" -}}
{{- $models := .Values.models -}}
{{- if $models -}}
{{- join "," $models -}}
{{- else -}}
{{- "" -}}
{{- end -}}
{{- end -}}