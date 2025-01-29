python := python3

define venvWrapper
	{\
	. .venv/bin/activate; \
	$1; \
	}
endef


install:
	@{ \
		echo "Setting up..."; \
		python3 -m venv .venv; \
		. .venv/bin/activate; \
		if [ -f requirements.txt ]; then \
			pip install -r requirements.txt; \
			echo "Installing dependencies...DONE"; \
		fi; \
	}

freeze:
	$(call venvWrapper, pip freeze > requirements.txt)

all: train


distribution:
	@$(call venvWrapper, python3 distribution.py $(arg))

augmentation:
	@$(call venvWrapper, python3 augmentation.py $(arg))

transformation:
	@$(call venvWrapper, python3 transformation.py $(arg))

train:
	@$(call venvWrapper, python3 train.py $(arg))

predict:
	@$(call venvWrapper, python3 predict.py $(arg))

clean:


fclean: clean
	@rm -rf .venv/ .venv/bin/ .venv/include/ .venv/lib/ .venv/lib64 .venv/pyvenv.cfg .venv/share/
	@rm -rf ./data

phony: install freeze process train predict