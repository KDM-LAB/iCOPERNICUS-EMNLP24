MOD_NAME=$1
MOD_CAT=$2
FOD=$3
get_prefix_before_underscore() {
    local SCREEN_NAME=$1
    local PREFIX=$(echo "$SCREEN_NAME" | cut -d'_' -f1)
    echo "$PREFIX"
}

# Function to run commands in a screen
run_commands() {
    SCREEN_NAME=$1
    PREFIX=$(get_prefix_before_underscore "$SCREEN_NAME")
    screen -dmS $SCREEN_NAME
    screen -S $SCREEN_NAME -X stuff ". venv/bin/activate"$'\n'
    screen -S $SCREEN_NAME -X stuff "cd pls"$'\n'
    screen -S $SCREEN_NAME -X stuff "cd $FOD"$'\n'
    echo "model=$MOD_NAME mod_cat=$MOD_CAT typer evaluation_script.py run generate-scores $PREFIX $MOD_CAT"
    screen -S $SCREEN_NAME -X stuff "model=$MOD_NAME mod_cat=$MOD_CAT typer evaluation_script.py run generate-scores $PREFIX $MOD_CAT"$'\n'
    screen -S $SCREEN_NAME -X detach
    sleep 1  # Adjust the sleep duration as needed
}

# Run commands in screens A, B, C, D, and E in parallel
run_commands bleu_$MOD_CAT &
run_commands JSD_$MOD_CAT &
run_commands meteor_$MOD_CAT &
run_commands rougeL_$MOD_CAT &
run_commands rougeSU4_$MOD_CAT &

model=$MOD_NAME mod_cat=$MOD_CAT typer evaluation_script.py run generate-scores $PREFIX $MOD_CAT
