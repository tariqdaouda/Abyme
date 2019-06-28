# Abyme (Fractals)

Abyme is a tool for writing Deep and Sophisticated (Training) Loops.

Training loops involve a lot cuisine:
* When to save a model
* What to print on screen
* When?
* What information capture for debugging
* In what format save them?
* At which periodicity?

With Abyme training loops are written as fractals that go deeper and deeper, allowing the user to dynamically plug events at *user-defined* steps. Sounds complicated but it actually makes everyting much simpler.

```python
    ground("dig",
        looper.setup(10)("iteration_start",
            data_looper(
                "iteration_end",
                trainer("end",
                    AB.PeriodicTrigger(100)("trigger",
                        pp_store.setup("last_loss")
                    ),
                    AB.PeriodicTrigger(1)("trigger",
                        AB.CSVStreamSaver(filemame="test.csv", prefix="DEMO_", select_fields=["last_loss"], pandas_to_csv_kwargs = {"sep": ";"})
                    )
                )
            )
        ).more("end",
            AB.PrintMessage("the real end")
        ).more("start",
            AB.PrintMessage("Training starts")
        )
    ).dig()```
