echo "copying"
cp -r /shared_volume/llm/Llama-2-7b-chat-hf .
echo "copied"
python -u llm_api.py parameters/llm_deployment_parameters.yaml
