
def plausibility_prompt(var1, var2):
    return (
        f"Is it plausible that {var1} may have a causal relationship with {var2}, either directly or through one or more intermediate variables? "
        "Plausibility refers to the extent to which a hypothesized relationship may be biologically, theoretically, or scientifically reasonable, based on existing knowledge. "
        "Output TRUE if it is plausible, FALSE if not."
    )


def temporality_prompt(var1, var2):
    return (
        f"Is there a causal relationship where a change in {var1} precedes and leads to a change in {var2}, either directly or through one or more intermediate variables? "
        f"That is, does {var1} temporally precede and influence {var2} in a way consistent with a directional causal link? "
        "Output TRUE if there is a causal relationship and FALSE if not."
    )


def causal_lit_prompt(var1, var2):
    return (
        "Given the above information/definition/your knowledge/given text, which of the following is the most likely:\n"
        f"A. Changing {var1} causes a change in {var2}, either directly or through one or more intermediate variables.\n"
        f"B. Changing {var2} causes a change in {var1}, either directly or through one or more intermediate variables.\n"
        f"C. There is no causal relationship between {var1} and {var2}.\n"
    )

def association_prompt(var1, var2):
    return (
        "Association refers to any statistical dependency between two variables, regardless of strength "
        "or directionality. That is, if knowledge of the value of one variable gives information about the "
        f"other, they are said to be associated. Is there an association between {var1} and {var2}? "
        "Output True if there exists a statistical association between the two variables, False, if not."

    )