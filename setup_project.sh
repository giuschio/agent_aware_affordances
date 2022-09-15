PROJECT_DIRECTORY=$(pwd)

echo "creating empty data folder"
mkdir -p 'src/data'

CONFIGURATION_FILE='.bashrc_aff'
CONFIGURATION_PATH="$PROJECT_DIRECTORY/$CONFIGURATION_FILE"
echo "cleaning previous configuration (if any)"
echo "" > "$CONFIGURATION_PATH"

SOURCE_FOLDER='src'
SOURCE_PATH="$PROJECT_DIRECTORY/$SOURCE_FOLDER"
echo "adding project source to environment variables"
echo "export AFFORDANCE_SOURCE=$SOURCE_PATH" >> ./.bashrc_aff
echo "exporting project source path: $SOURCE_PATH"
echo "export PYTHONPATH=$PYTHONPATH:$SOURCE_PATH" >> ./.bashrc_aff

VENV_FILE='affordances_venv/bin/activate'
VENV_PATH="$PROJECT_DIRECTORY/$VENV_FILE"
echo "aliasing source-aff as source $VENV_PATH"
echo "alias source-aff='source $VENV_PATH'" >> ./.bashrc_aff
echo "source-aff" >> ./.bashrc_aff

echo "sourcing virtual environment"
source "$VENV_PATH"

echo "please source .bashrc_aff before running each python script"
