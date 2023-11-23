#! /bin/bash
# This is for solving some clangd includePaths Issue.
# See https://github.com/Sarcasm/compdb#generate-a-compilation-database-with-header-files
compdb -p build/ list > compile_commands.json
