# Recipes for building ubuntu images.  

.PHONY:
shell:
	sudo docker compose -f "./dockers/docker-compose.yml" run ubuntu_shell_2204 
	
.PHONY:
shell_build:
	sudo docker compose  -f "./dockers/docker-compose.yml"  build ubuntu_shell_2204
