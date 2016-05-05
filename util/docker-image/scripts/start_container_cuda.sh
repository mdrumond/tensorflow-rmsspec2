dir_resolve()
{
    cd "$1" 2>/dev/null || return $?  # cd to desired directory; if fail, quell any error messages but return exit status
    echo "`pwd -P`" # output full, link-resolved path
}


DATA_PATH="`dir_resolve \"../data\"`"
SCRIPTS_PATH="`dir_resolve \"../scripts\"`"

nvidia-docker run -d \
              -p 0.0.0.0:2323:22 \
              -p 0.0.0.0:7007:6006 \
              --name tensorflow \
              -v $DATA_PATH:/data \
              -v $SCRIPTS_PATH:/scripts \
              -t tensorflow-ssh
